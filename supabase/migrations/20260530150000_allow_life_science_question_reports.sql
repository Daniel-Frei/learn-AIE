alter table public.question_reports
  drop constraint if exists question_reports_topic_check;

alter table public.question_reports
  add constraint question_reports_topic_check
  check (topic in ('RL', 'DL', 'NLP', 'Math', 'Life Science'));
