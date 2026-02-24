

I want you to create 40 questions so that I can learn the lecture (see transcript below and attached pdf for the slides). As this often requires a lot of tokens, I think it's best to break this up into two batches (provide 20 questions, then in a next step we can add 20 more questions)

I already created lib\lectures\MIT 15.773 Hands-On Deep Learning Spring 2024\L1_ Introduction to Neural Networks and Deep Learning.ts for those questions. Note that this is referring to a lecture but the same requirements (below) as for chapters apply.

Requirements:
- Same format as we already used with four potential answers/options, the options/answer should be true or false (more than one answer can be true or false), explanation.
- each question with four options
- the order of the options / answers is random (so that I don't accidentally remember the order), so basically we can use the current implementation. This also means that the order of the answers shouldn't matter (neither how they are beind displayed to the user nor the order of the user ticking the boxes; e.g. don't have answers that say "First..." and then in another answer "Then.." as the user might not see the answers in that order)
- questions can cover anything from simple terminology questions (e.g. "what does RNN stand for?") and basic concepts to more difficult and complex questions. The goal is to test (and practice) the understanding of this chapter (or lecture transcript). It should also include math related questions (related to the equations in the chapter or lectures, if there are any). You can include math in questions and/or answers.
- Don't include any questions about logistics or admin of lectures (e.g. when the exam will take place) or books (e.g. where students can access online resources).
- the goal of the question isn't to literally ask about the chapter (or lecture) but instead test / practice the knowledge in the chapter / lecture. So imagine if someone tried to answer the questions but didn't read the chapter (didn't attend the lecture) but still has the same knowledge as what was discussed. They should be able to answer the questions and not fail due to vague references to the chapter / lecture (e.g. "the equation in the chapter"). Again, don't directly refer to the chapter or lecture in questions.
- easy questions should focus on terminology, simple concept, key concepts including concepts might not be explicitly explained in the chapter but important to understand it... Mastering easy questions should make it easier to understand concepts in medium and hard question. e.g. some medium and hard questions might mention "logits" but what logits are is never asked in an easy question.
- easy questions can also cover topics where there are no medium or hard questions for.
- Questions should cover all concepts and terms that what were discussed in the chapter.
- Write out acronyms.
- it's important that you directly include math in questions when it's heavily covered in the lecture / chapter. Don't avoid mathematical notations if they are covered in the lecture / chapter.
- Avoid answer / options that are too easy to guess without having knowledge of the topics. E.g. absolute terms like "always", "all cases", "every time", "impossible" and "never" usually make it very likely that a statement is wrong. Try to be smart about answer options, so that incorrect answers still seem somewhat plausible.
- Options / answers should be independent of each others (not reference each other or not require a specific order).
- The answer patterns should be equaly destributed: 25% of all questions should where all answers are true, 25% should be where three answers are true, 25% should be where two answers are true, 25% should be where one answer is true. Note that by true I am referring to the user ticking the box, the statement in the answers doesn't have to be true, e.g. the question could be "which of the following statements are false?" and then the answer "RNNs use transformers" would be a true answer even though the statement by itself is wrong.
- There should roughly be the same amount of easy, medium and hard questions.
- The difficulty (amount of easy, medium and hard) and answer type (25% each) should roughly be true (sometimes it's not possible if the total question number isn't dividable accordingly)
- When in doubt, it's better for the descriptions (and the explanation) to be a bit longer. They should help the user understand the questions and answers better, so a bit of extra information is usually desired.
- Especially the explanations (which the user sees after answering the question) should really explain the answers (at least two sentences long). The explanation should explain the issues in simple terms, so that the user can learn from it.
- provide the questions as code (see example)


Question Authoring Guide for Math & Formatting

For math, wrap LaTeX in:

Inline math: \( ... \)
Example: The ratio is \( \rho_t = \frac{\pi_{\theta'}(a_t\mid s_t)}{\pi_{\theta}(a_t\mid s_t)} \).

Block math: \[ ... \]
Example:
The return is
\[ G_t = \sum_{t'=t}^T r_{t'} \]

Notes:

Always escape backslashes in TS/JS strings: write \\( ... \\) and \\[ ... \\].
Keep each prompt/option/explanation as a single string. The renderer will handle multiple sentences and line breaks.
Use standard LaTeX for fractions, sums, subscripts, and Greek letters: \frac{a}{b}, \sum, _t, \theta, \pi.

Example:

prompt: "In off-policy policy gradient, what does the importance ratio \\(\\rho_t\\) represent?",
options: [
  { text: "\\(\\rho_t = \\frac{\\pi_{\\theta'}(a_t\\mid s_t)}{\\pi_{\\theta}(a_t\\mid s_t)}\\)", isCorrect: true },
  { text: "\\(\\rho_t = \\frac{\\pi_{\\theta}(a_t\\mid s_t)}{\\pi_{\\theta'}(a_t\\mid s_t)}\\)", isCorrect: false },
  { text: "\\(\\rho_t\\) is the baseline \\(b\\).", isCorrect: false },
  { text: "\\(\\rho_t\\) is the return \\(G_t = \\sum_{t'=t}^T r_{t'}\\).", isCorrect: false },
],
explanation:
  "Off-policy PG reweights samples from behavior policy \\(\\pi_\\theta\\) to estimate gradients for target policy \\(\\pi_{\\theta'}\\). The per-step ratio is \\(\\rho_t = \\pi_{\\theta'}(a_t\\mid s_t) / \\pi_{\\theta}(a_t\\mid s_t)\\)."
