alter table public.question_reports
  drop constraint if exists question_reports_topic_check;

alter table public.question_reports
  drop constraint if exists question_reports_topic_present_check;

alter table public.question_reports
  add constraint question_reports_topic_present_check
  check (char_length(btrim(topic)) between 1 and 200);
