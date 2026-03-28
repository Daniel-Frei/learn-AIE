-- Shared quiz data schema for Supabase Postgres.
-- Apply this in the Supabase SQL editor before using the shared-storage routes.

create table if not exists public.participants (
  participant_id text primary key,
  rating double precision not null,
  rd double precision not null,
  sigma double precision not null,
  last_updated_at bigint not null,
  games_played integer not null default 0,
  legacy_migrated_at timestamptz null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.question_ratings (
  question_id text primary key,
  rating double precision not null,
  rd double precision not null,
  sigma double precision not null,
  last_updated_at bigint not null,
  games_played integer not null default 0,
  legacy_correct integer not null default 0,
  legacy_wrong integer not null default 0,
  label text null check (label in ('easy', 'medium', 'hard')),
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.answer_attempts (
  attempt_id text primary key,
  participant_id text not null references public.participants(participant_id) on delete cascade,
  question_id text not null,
  label text null check (label in ('easy', 'medium', 'hard')),
  is_correct boolean not null,
  answered_at timestamptz not null default timezone('utc', now()),
  source text not null check (source in ('live', 'migration'))
);

create index if not exists answer_attempts_participant_idx
  on public.answer_attempts(participant_id, answered_at desc);

create index if not exists answer_attempts_question_idx
  on public.answer_attempts(question_id, answered_at desc);

create table if not exists public.question_reports (
  id text primary key,
  participant_id text not null references public.participants(participant_id) on delete cascade,
  question_id text not null,
  comment text not null,
  reported_at timestamptz not null,
  source_id text not null,
  source_label text not null,
  series_id text not null,
  series_label text not null,
  topic text not null check (topic in ('RL', 'DL', 'NLP', 'Math')),
  prompt text not null
);

create index if not exists question_reports_question_idx
  on public.question_reports(question_id, reported_at desc);
