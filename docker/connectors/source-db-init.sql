-- Local fake source for connector sync: seeded notices and a role that can only read them.
CREATE TABLE notices (
    id integer PRIMARY KEY,
    title text NOT NULL,
    body text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO notices (id, title, body) VALUES
    (1, 'Water shutoff', 'Water is off on Monday from 9 to 11 for pipe repairs.'),
    (2, 'Lift service', 'The lift is serviced on Wednesday morning.'),
    (3, 'Recycling', 'Glass goes in the green bin behind block B.');

-- Fake-only credential on an internal network; never reuse it for a real source.
CREATE ROLE notices_reader LOGIN PASSWORD 'local-fake-source-only';
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO notices_reader;
GRANT SELECT ON notices TO notices_reader;
ALTER ROLE notices_reader SET default_transaction_read_only = on;
