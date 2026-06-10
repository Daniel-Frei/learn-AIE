alter table public.question_reports
  add column if not exists status text not null default 'open';

alter table public.question_reports
  add column if not exists resolved_at timestamptz null;

alter table public.question_reports
  add column if not exists resolution_note text null;

alter table public.question_reports
  drop constraint if exists question_reports_status_check;

alter table public.question_reports
  add constraint question_reports_status_check
  check (status in ('open', 'resolved'));

alter table public.question_reports
  drop constraint if exists question_reports_resolution_consistency_check;

alter table public.question_reports
  add constraint question_reports_resolution_consistency_check
  check (
    (status = 'resolved' and resolved_at is not null)
    or (status = 'open' and resolved_at is null)
  );

alter table public.question_reports
  drop constraint if exists question_reports_resolution_note_length_check;

alter table public.question_reports
  add constraint question_reports_resolution_note_length_check
  check (char_length(coalesce(resolution_note, '')) <= 2000);

create index if not exists question_reports_status_question_idx
  on public.question_reports(status, question_id, reported_at desc);

grant select (id, question_id, status) on table public.question_reports to authenticated;
