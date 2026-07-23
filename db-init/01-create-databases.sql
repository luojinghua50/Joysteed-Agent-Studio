-- Create additional databases beyond POSTGRES_DB (agent_core).
-- Runs only on first cluster init (empty pg_data volume).
SELECT 'CREATE DATABASE agent_rag'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'agent_rag')\gexec

SELECT 'CREATE DATABASE langfuse'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')\gexec
