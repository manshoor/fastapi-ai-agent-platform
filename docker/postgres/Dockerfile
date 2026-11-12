# PostgreSQL 19 rc1 plus the pgvector extension used by RAG chunk search.
# The PGDG package depends only on postgresql-19, so the server stays 19rc1.
FROM postgres:19rc1-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-19-pgvector=0.8.7-1.pgdg12+1 \
    && rm -rf /var/lib/apt/lists/* \
    && test -f /usr/share/postgresql/19/extension/vector.control
