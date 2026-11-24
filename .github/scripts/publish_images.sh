#!/usr/bin/env bash
# Publish the smoke-tested image archives as one multi-architecture tag per image.
#
#   publish_images.sh <registry prefix> <commit> <archive folder>
#   publish_images.sh ghcr.io/manshoor 0123...cdef archives
#
#   images-amd64.tar --+
#                      +--> local folder: two manifests + one index --> registry :sha-<commit>
#   images-arm64.tar --+
#
# Reads bytes only: it never builds or runs an image. Prints `<image>=<index digest>`
# lines on stdout and progress on stderr.
set -euo pipefail

readonly REPOSITORY=fastapi-ai-agent-platform
readonly ARCHIVE_TAG=ci
readonly IMAGES=(app postgres simulator)
readonly PLATFORMS=(amd64 arm64)
readonly MANIFEST_TYPE=application/vnd.docker.distribution.manifest.v2+json
readonly INDEX_TYPE=application/vnd.docker.distribution.manifest.list.v2+json

fail() {
  echo "publish_images: $1" >&2
  exit 1
}

digest_of() {
  echo "sha256:$(sha256sum "$1" | cut -d ' ' -f 1)"
}

repository_of() {
  if [[ "$1" == app ]]; then
    echo "$REPOSITORY"
  else
    echo "$REPOSITORY-$1"
  fi
}

if [[ $# -ne 3 ]]; then
  fail "expected <registry prefix> <commit> <archive folder>"
fi
prefix=$1
commit=$2
archives=$3
if [[ ! "$commit" =~ ^[0-9a-f]{40}$ ]]; then
  fail "commit must be 40 lowercase hex characters"
fi

# Plain HTTP is for the throwaway registry on this machine only.
tls_verify=true
case "$prefix" in
  localhost:* | 127.0.0.1:*) tls_verify=false ;;
esac

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# Write one platform's manifest and layers into the image folder. The Docker format
# keeps the config bytes as they are; converting to OCI would rewrite them.
add_platform() {
  local image=$1 platform=$2
  local archive="$archives/images-$platform.tar"
  local reference single="$work/$image-$platform" folder="$work/$image"
  local config_path config manifest
  reference="$(repository_of "$image"):$ARCHIVE_TAG"

  # The config names what was smoke-tested: check it is this commit and platform.
  config_path=$(tar -xOf "$archive" manifest.json | jq -er --arg tag "$reference" \
    '[.[] | select((.RepoTags // []) | index($tag))] | select(length == 1) | .[0].Config') ||
    fail "$archive does not hold exactly one $reference"
  config="$work/$image-$platform.config"
  tar -xOf "$archive" "$config_path" > "$config"
  jq -e --arg arch "$platform" --arg commit "$commit" '
    .os == "linux" and .architecture == $arch
    and .config.Labels["org.opencontainers.image.revision"] == $commit' "$config" > /dev/null ||
    fail "$reference is not a linux/$platform image of $commit"

  # Compress now, so the digests below are final.
  skopeo copy --quiet --format v2s2 --dest-compress \
    "docker-archive:$archive:$reference" "dir:$single" >&2
  manifest=$(digest_of "$single/manifest.json")
  mv "$single/manifest.json" "$folder/${manifest#sha256:}.manifest.json"
  mv "$single/version" "$folder/version"
  mv "$single"/* "$folder/"

  jq -n --arg type "$MANIFEST_TYPE" --arg digest "$manifest" --arg arch "$platform" \
    --argjson size "$(wc -c < "$folder/${manifest#sha256:}.manifest.json")" '
    {mediaType: $type, size: $size, digest: $digest,
     platform: {architecture: $arch, os: "linux"}}' >> "$work/$image.entries"
}

# Assemble every image before pushing any, so a bad archive publishes nothing.
for image in "${IMAGES[@]}"; do
  mkdir "$work/$image"
  for platform in "${PLATFORMS[@]}"; do
    echo "Reading $image $platform" >&2
    add_platform "$image" "$platform"
  done
  jq -s --arg type "$INDEX_TYPE" '{schemaVersion: 2, mediaType: $type, manifests: .}' \
    "$work/$image.entries" > "$work/$image/manifest.json"
done

# Push both platforms by digest and the index by tag, then read everything back.
for image in "${IMAGES[@]}"; do
  target="$prefix/$(repository_of "$image")"
  tag="$target:sha-$commit"
  echo "Pushing $tag" >&2
  skopeo copy --quiet --all --retry-times 3 --preserve-digests --dest-tls-verify="$tls_verify" \
    "dir:$work/$image" "docker://$tag" >&2

  index=$(digest_of "$work/$image/manifest.json")
  skopeo inspect --raw --tls-verify="$tls_verify" "docker://$tag" > "$work/pushed"
  if [[ "$(digest_of "$work/pushed")" != "$index" ]]; then
    fail "$tag is not the index that was built"
  fi
  for platform in "${PLATFORMS[@]}"; do
    manifest=$(jq -er --arg arch "$platform" \
      '.manifests[] | select(.platform.architecture == $arch) | .digest' "$work/pushed")
    pushed=$(skopeo inspect --raw --tls-verify="$tls_verify" "docker://$target@$manifest" |
      jq -er .config.digest)
    if [[ "$pushed" != "$(digest_of "$work/$image-$platform.config")" ]]; then
      fail "$target $platform config differs from the archive"
    fi
  done
  echo "$image=$index"
done
