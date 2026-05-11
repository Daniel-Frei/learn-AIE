create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text null,
  display_name text null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

alter table public.answer_attempts
  add column if not exists elapsed_ms integer not null default 0,
  add column if not exists mistake_count integer not null default 0;

alter table public.profiles enable row level security;
alter table public.participants enable row level security;
alter table public.question_ratings enable row level security;
alter table public.answer_attempts enable row level security;
alter table public.question_reports enable row level security;

drop policy if exists "profiles_select_own" on public.profiles;
create policy "profiles_select_own"
  on public.profiles for select
  to authenticated
  using (id = auth.uid());

drop policy if exists "profiles_insert_own" on public.profiles;
create policy "profiles_insert_own"
  on public.profiles for insert
  to authenticated
  with check (id = auth.uid());

drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_update_own"
  on public.profiles for update
  to authenticated
  using (id = auth.uid())
  with check (id = auth.uid());

drop policy if exists "participants_select_own" on public.participants;
create policy "participants_select_own"
  on public.participants for select
  to authenticated
  using (participant_id = auth.uid()::text);

drop policy if exists "participants_insert_own" on public.participants;
create policy "participants_insert_own"
  on public.participants for insert
  to authenticated
  with check (participant_id = auth.uid()::text);

drop policy if exists "participants_update_own" on public.participants;
create policy "participants_update_own"
  on public.participants for update
  to authenticated
  using (participant_id = auth.uid()::text)
  with check (participant_id = auth.uid()::text);

drop policy if exists "question_ratings_select_authenticated" on public.question_ratings;
create policy "question_ratings_select_authenticated"
  on public.question_ratings for select
  to authenticated
  using (true);

drop policy if exists "question_ratings_insert_authenticated" on public.question_ratings;
create policy "question_ratings_insert_authenticated"
  on public.question_ratings for insert
  to authenticated
  with check (true);

drop policy if exists "question_ratings_update_authenticated" on public.question_ratings;
create policy "question_ratings_update_authenticated"
  on public.question_ratings for update
  to authenticated
  using (true)
  with check (true);

drop policy if exists "answer_attempts_select_own" on public.answer_attempts;
create policy "answer_attempts_select_own"
  on public.answer_attempts for select
  to authenticated
  using (participant_id = auth.uid()::text);

drop policy if exists "answer_attempts_insert_own" on public.answer_attempts;
create policy "answer_attempts_insert_own"
  on public.answer_attempts for insert
  to authenticated
  with check (participant_id = auth.uid()::text);

drop policy if exists "question_reports_select_authenticated" on public.question_reports;
create policy "question_reports_select_authenticated"
  on public.question_reports for select
  to authenticated
  using (true);

drop policy if exists "question_reports_insert_own" on public.question_reports;
create policy "question_reports_insert_own"
  on public.question_reports for insert
  to authenticated
  with check (participant_id = auth.uid()::text);
