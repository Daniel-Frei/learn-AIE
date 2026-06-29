import { expect, test, type Page } from "@playwright/test";

const ROUTE_TRANSITION_TIMEOUT_MS = 15000;

function trackSelectionPermissionErrors(page: Page) {
  const errors: string[] = [];
  const selectionPermissionPattern =
    /Permission denied to access property "(?:__reactFiber|correspondingUseElement)/i;

  page.on("console", (message) => {
    if (
      message.type() === "error" &&
      selectionPermissionPattern.test(message.text())
    ) {
      errors.push(message.text());
    }
  });
  page.on("pageerror", (error) => {
    if (selectionPermissionPattern.test(error.message)) {
      errors.push(error.message);
    }
  });

  return errors;
}

test("lists available learning courses", async ({ page }) => {
  await page.goto("/learn");

  await expect(
    page.getByRole("heading", { name: /choose a course/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /stanford cme295 transformers & llms/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /crash course probability/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /clinical trials crash course/i }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: /ai agents/i })).toBeVisible();
  await expect(
    page.getByRole("link", { name: /crash course medicine/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /text into transformers/i }),
  ).toHaveCount(0);
});

test("lists learning experiences for a selected course", async ({ page }) => {
  await page.goto("/learn/crash-course-probability");

  await expect(
    page.getByRole("heading", { name: /crash course probability/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /likelihood, loss, softmax/i }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: /rl over time/i })).toBeVisible();
  await expect(
    page.getByRole("link", { name: /generation sampling lab/i }),
  ).toBeVisible();

  await page.getByRole("link", { name: /rl over time/i }).click();
  await expect(page).toHaveURL(
    /\/learn\/crash-course-probability\/crash-probability-l4$/,
    {
      timeout: ROUTE_TRANSITION_TIMEOUT_MS,
    },
  );
});

test("lists the AI Agents memory learning page for its course", async ({
  page,
}) => {
  await page.goto("/learn/ai-agents");

  await expect(
    page.getByRole("heading", { name: /^AI Agents$/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /memory system workbench/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /memory evaluation console/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /atommem pipeline debugger/i }),
  ).toBeVisible();
});

test("lists the standalone Medicine learning page for its course", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-medicine");

  await expect(
    page.getByRole("heading", { name: /crash course medicine/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", {
      name: /what medicine is/i,
    }),
  ).toBeVisible();

  await page.getByRole("link", { name: /what medicine is/i }).click();
  await expect(page).toHaveURL(/\/learn\/crash-course-medicine\/lecture-1$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
});

test("labels and sorts course learning experiences by source sequence", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295");

  const courseCards = page.locator(
    'section[aria-label="Stanford CME295 Transformers & LLMs learning experiences"] a',
  );
  await expect(courseCards).toHaveCount(9);

  const cardTexts = await courseCards.allTextContents();
  expect(cardTexts[0]).toContain("Lecture 1");
  expect(cardTexts[1]).toContain("Lecture 2");
  expect(cardTexts[2]).toContain("Lecture 3");
  expect(cardTexts[3]).toContain("Lecture 4");
  expect(cardTexts[4]).toContain("Lecture 5");
  expect(cardTexts[5]).toContain("Lecture 6");
  expect(cardTexts[5]).toContain("Reasoning Control Bench");
  expect(cardTexts[6]).toContain("Lecture 7");
  expect(cardTexts[6]).toContain("RAG, Tools, Agents Studio");
  expect(cardTexts[7]).toContain("Lecture 8");
  expect(cardTexts[7]).toContain("LLM Evaluation Studio");
  expect(cardTexts[8]).toContain("Lecture 9");
  expect(cardTexts[8]).toContain("Course Recap Synthesis");
  expect(cardTexts.join("\n")).not.toMatch(/15 min|16 min|18 min|19 min/i);
  expect(cardTexts.join("\n")).not.toMatch(
    /Introductory NLP with ML basics|After CME295 Lecture/i,
  );
});

test("opens the standalone Stanford CME295 Lecture 6 page from the course page", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295");

  await page.getByRole("link", { name: /reasoning control bench/i }).click();
  await expect(page).toHaveURL(/\/learn\/stanford-cme295\/lecture-6$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /run a reasoning model like a controlled experiment/i,
    }),
  ).toBeVisible();
});

test("opens the standalone Stanford CME295 Lecture 7 page from the course page", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295");

  await page.getByRole("link", { name: /rag, tools, agents studio/i }).click();
  await expect(page).toHaveURL(/\/learn\/stanford-cme295\/lecture-7$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /connect language models to systems they can inspect and act through/i,
    }),
  ).toBeVisible();
});

test("opens the standalone Stanford CME295 Lecture 8 page from the course page", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295");

  await page.getByRole("link", { name: /llm evaluation studio/i }).click();
  await expect(page).toHaveURL(/\/learn\/stanford-cme295\/lecture-8$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /build the evaluation console before improving the model/i,
    }),
  ).toBeVisible();
});

test("opens the Stanford CME295 Lecture 9 synthesis page from the course page", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295");

  await page.getByRole("link", { name: /course recap synthesis/i }).click();
  await expect(page).toHaveURL(/\/learn\/stanford-cme295\/cme295-lect9$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /reconstruct the transformer course from the recap/i,
    }),
  ).toBeVisible();
});

test("keeps legacy direct learning source URLs working", async ({ page }) => {
  await page.goto("/learn/crash-probability-l3");

  await expect(
    page.getByRole("heading", {
      name: /make the observed answer more probable/i,
    }),
  ).toBeVisible();
});

test("renders the Stanford CME295 Lecture 1 learning page and supports interactions", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect1");

  await expect(
    page.getByRole("heading", { name: /follow text into a transformer/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /tokenization tradeoff/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*QK/)).toHaveCount(0);

  const tokenizationExplorer = page.getByTestId("tokenization-explorer");
  await tokenizationExplorer.getByRole("button", { name: /^Word$/i }).click();
  await expect(tokenizationExplorer.getByRole("status")).toHaveText(
    /unknown token/i,
  );

  const attentionExplorer = page.getByTestId("attention-focus-explorer");
  await attentionExplorer
    .getByRole("button", { name: /query: reading/i })
    .click();
  await expect(attentionExplorer.getByText(/top focus: bear/i)).toBeVisible();

  const qkvCheck = page.getByTestId("qkv-check");
  await qkvCheck.getByRole("button", { name: /values are compared/i }).click();
  await expect(qkvCheck.getByRole("status")).toHaveText(/not yet/i);

  await qkvCheck.getByRole("button", { name: /queries are compared/i }).click();
  await expect(qkvCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the Stanford CME295 Lecture 2 learning page and supports interactions", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect2");

  await expect(
    page.getByRole("heading", { name: /tune the transformer upgrade knobs/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /position and attention design bench/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /bert input and fine-tuning lab/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*PE/)).toHaveCount(0);

  const positionBench = page.getByTestId("position-attention-bench");
  await positionBench.getByRole("button", { name: /^ALiBi$/i }).click();
  await expect(positionBench.getByRole("status")).toHaveText(
    /deterministic linear distance penalty/i,
  );

  const efficiencyLab = page.getByTestId("attention-efficiency-lab");
  await efficiencyLab.getByRole("button", { name: /^MQA$/i }).click();
  await expect(efficiencyLab.getByRole("status")).toHaveText(
    /all heads share keys and values/i,
  );

  const bertLab = page.getByTestId("bert-input-lab");
  await bertLab.getByRole("button", { name: /token task/i }).click();
  await bertLab.getByRole("button", { name: /^Output$/i }).click();
  await expect(bertLab.getByRole("status")).toHaveText(
    /each non-special token output/i,
  );

  const positionCheck = page.getByTestId("position-extrapolation-check");
  await positionCheck
    .getByRole("button", { name: /softmax cannot normalize/i })
    .click();
  await expect(positionCheck.getByRole("status")).toHaveText(/not yet/i);

  await positionCheck
    .getByRole("button", { name: /no learned position vectors/i })
    .click();
  await expect(positionCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the Stanford CME295 Lecture 3 learning page and supports interactions", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect3");

  await expect(
    page.getByRole("heading", {
      name: /follow one request through llm inference/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /trace one request through the llm runtime/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /place each technique before comparing tradeoffs/i,
    }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*P\(t_i\)/)).toHaveCount(0);

  const runtimeLab = page.getByTestId("llm-runtime-lab");
  await runtimeLab.getByRole("button", { name: /next-token policy/i }).click();
  await runtimeLab.getByRole("button", { name: /Guided JSON/i }).click();
  await expect(runtimeLab.getByRole("status")).toHaveText(/external grammar/i);

  await runtimeLab.getByRole("button", { name: /MoE/i }).click();
  await runtimeLab.getByRole("button", { name: /^Collapsed$/i }).click();
  await expect(runtimeLab.getByRole("status")).toHaveText(
    /Most tokens route to one expert/i,
  );

  await runtimeLab.getByRole("button", { name: /Cache/i }).click();
  await runtimeLab.getByRole("button", { name: /^PagedAttention$/i }).click();
  await expect(runtimeLab.getByRole("status")).toHaveText(
    /non-contiguous pages/i,
  );

  await runtimeLab.getByRole("button", { name: /Accelerate/i }).click();
  await runtimeLab.getByRole("button", { name: /^MTP$/i }).click();
  await expect(runtimeLab.getByRole("status")).toHaveText(
    /extra heads predict future positions/i,
  );

  const routingCheck = page.getByTestId("routing-collapse-check");
  await routingCheck
    .getByRole("button", { name: /top-p sampling instead of beam search/i })
    .click();
  await expect(routingCheck.getByRole("status")).toHaveText(/not yet/i);

  await routingCheck
    .getByRole("button", { name: /router has collapsed/i })
    .click();
  await expect(routingCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the Stanford CME295 Lecture 4 learning page and supports interactions", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect4");

  await expect(
    page.getByRole("heading", { name: /run the llm training pipeline/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /follow one model from raw continuation to tuned behavior/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /match the memory object to the optimization lever/i,
    }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*C_\{\\mathrm/)).toHaveCount(0);

  const lifecycle = page.getByTestId("training-lifecycle-workbench");
  await lifecycle.getByRole("button", { name: /stage 3\s+sft/i }).click();
  await expect(lifecycle.getByRole("status")).toHaveText(
    /instruction-tuned model/i,
  );

  const scalingLab = page.getByTestId("scaling-budget-lab");
  await scalingLab
    .getByRole("button", { name: /gpt-3-style example/i })
    .click();
  await expect(scalingLab.getByRole("status")).toHaveText(/undertrained/i);

  const memoryLab = page.getByTestId("memory-optimization-lab");
  await memoryLab.getByRole("button", { name: /^FlashAttention$/i }).click();
  await expect(memoryLab.getByRole("status")).toHaveText(/exact attention/i);

  const loraWorkbench = page.getByTestId("lora-workbench");
  await loraWorkbench.getByRole("button", { name: /^QLoRA$/i }).click();
  await expect(loraWorkbench.getByRole("status")).toHaveText(
    /frozen W0 is stored quantized/i,
  );

  const flashCheck = page.getByTestId("flashattention-check");
  await flashCheck
    .getByRole("button", { name: /approximates attention/i })
    .click();
  await expect(flashCheck.getByRole("status")).toHaveText(/not yet/i);
  await flashCheck
    .getByRole("button", { name: /reduces slow hbm reads/i })
    .click();
  await expect(flashCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the Stanford CME295 Lecture 5 learning page and supports preference-tuning labs", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect5");

  await expect(
    page.getByRole("heading", {
      name: /tune preferences without chasing the proxy/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /turn a vague complaint into a training example/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /reward modeling turns pairs into score differences/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /read the alignment pipeline before the algorithms/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /map language generation into rl before using rlhf/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /ppo means proximal policy optimization/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /dpo compares what changed relative to the reference/i,
    }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*P\(y_w/)).toHaveCount(0);
  await expect(page.getByText(/State s_t/)).toHaveCount(0);
  await expect(page.getByText(/Action a_t/)).toHaveCount(0);
  await expect(page.getByText(/pi\(a_t \| s_t\)/)).toHaveCount(0);
  await expect(page.getByText(/Advantage A_t/)).toHaveCount(0);

  const preferenceStudio = page.getByTestId("preference-data-studio");
  await preferenceStudio.getByRole("button", { name: /^Pointwise/i }).click();
  await expect(preferenceStudio.getByRole("status")).toHaveText(
    /absolute 0\.9/i,
  );
  await preferenceStudio.getByRole("button", { name: /^Pairwise/i }).click();
  await expect(preferenceStudio.getByRole("status")).toHaveText(
    /choosing the better of two outputs/i,
  );

  const rewardLab = page.getByTestId("bradley-terry-lab");
  await rewardLab.getByRole("button", { name: /equal rewards/i }).click();
  await expect(rewardLab.getByRole("status")).toHaveText(
    /0\.50 preference probability/i,
  );

  const ppoLab = page.getByTestId("ppo-clip-lab");
  await expect(ppoLab.getByText(/0\.30 \/ 0\.20 = 1\.50/)).toBeVisible();
  await expect(ppoLab.getByText(/min\(3\.00, 2\.40\) = 2\.40/)).toBeVisible();
  await ppoLab.getByRole("button", { name: /negative advantage/i }).click();
  await expect(ppoLab.getByRole("status")).toHaveText(
    /limits how far probability can drop/i,
  );

  const bestOfNLab = page.getByTestId("best-of-n-lab");
  await expect(
    bestOfNLab.getByText(/1 - 0\.70\^4 = 0\.76/).first(),
  ).toBeVisible();
  await bestOfNLab.getByRole("button", { name: /N=8/i }).click();
  await expect(bestOfNLab.getByRole("status")).toHaveText(/roughly 8x/i);
  await expect(
    bestOfNLab.getByText(/1 - 0\.70\^8 = 0\.94/).first(),
  ).toBeVisible();

  const dpoLab = page.getByTestId("dpo-logit-lab");
  await expect(dpoLab.getByText(/0\.10 \* 1\.50 = 0\.15/)).toBeVisible();
  await dpoLab.getByRole("button", { name: /policy favors rejected/i }).click();
  await expect(dpoLab.getByRole("status")).toHaveText(/goes below zero/i);

  const rewardHackingCheck = page.getByTestId("reward-hacking-check");
  await rewardHackingCheck
    .getByRole("button", { name: /learned new facts/i })
    .click();
  await expect(rewardHackingCheck.getByRole("status")).toHaveText(/not yet/i);
  await rewardHackingCheck
    .getByRole("button", { name: /exploited a proxy reward/i })
    .click();
  await expect(rewardHackingCheck.getByRole("status")).toHaveText(/correct/i);

  const dpoCheck = page.getByTestId("dpo-check");
  await dpoCheck
    .getByRole("button", { name: /no longer needs preference data/i })
    .click();
  await expect(dpoCheck.getByRole("status")).toHaveText(/not yet/i);
  await dpoCheck
    .getByRole("button", { name: /preference pairs and a frozen reference/i })
    .click();
  await expect(dpoCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the standalone Stanford CME295 Lecture 6 page and supports reasoning labs", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/lecture-6");

  await expect(
    page.getByRole("heading", {
      name: /run a reasoning model like a controlled experiment/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /reasoning means solving through intermediate steps/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /pass@k answers a different question than pass@1/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /grpo grades a response against its sampled group/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /ppo and grpo share update control but differ in the baseline/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /start reasoning questions/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*A_i/)).toHaveCount(0);

  const primer = page.getByTestId("reasoning-primer");
  await primer.getByRole("button", { name: /direct answer/i }).click();
  await expect(primer.getByRole("status")).toHaveText(/little evidence/i);
  await primer
    .getByRole("button", { name: /hidden trace with summary/i })
    .click();
  await expect(primer.getByRole("status")).toHaveText(/summary/i);

  const capabilityLab = page.getByTestId("capability-bottleneck-lab");
  await page.getByRole("button", { name: /course-code lookup/i }).click();
  await capabilityLab.getByRole("button", { name: /^Reasoning$/i }).click();
  await expect(capabilityLab.getByRole("status")).toHaveText(/try again/i);
  await capabilityLab
    .getByRole("button", { name: /knowledge lookup/i })
    .click();
  await expect(capabilityLab.getByRole("status")).toHaveText(
    /correct bottleneck/i,
  );

  const benchmarkLab = page.getByTestId("benchmark-lab");
  await benchmarkLab.getByRole("button", { name: /k=8/i }).click();
  await expect(benchmarkLab.getByRole("status")).toHaveText(
    /1 - C\(6, 8\) \/ C\(10, 8\) = 1 - 0 \/ 45 = 1\.00/i,
  );
  await expect(
    benchmarkLab.getByText(/cons@k samples several answers/i),
  ).toBeVisible();
  await benchmarkLab.getByRole("button", { name: /split vote/i }).click();
  await expect(benchmarkLab.getByText(/weak plurality/i)).toBeVisible();

  const budgetLab = page.getByTestId("reasoning-budget-lab");
  await budgetLab.getByRole("button", { name: /simple lookup/i }).click();
  await expect(budgetLab.getByRole("status")).toHaveText(
    /extra thinking mostly adds latency/i,
  );

  const rewardLab = page.getByTestId("verifiable-reward-lab");
  await rewardLab.getByRole("button", { name: /open-ended advice/i }).click();
  await expect(rewardLab.getByRole("status")).toHaveText(
    /not directly verifiable/i,
  );

  const grpoLab = page.getByTestId("grpo-group-lab");
  await expect(grpoLab.getByText(/GRPO objective skeleton/i)).toBeVisible();
  await grpoLab.getByRole("button", { name: /wrong but long/i }).click();
  await expect(grpoLab.getByRole("status")).toHaveText(
    /should not be rewarded for length/i,
  );

  const comparisonLab = page.getByTestId("ppo-grpo-comparison-lab");
  await comparisonLab.getByRole("button", { name: /^PPO$/i }).click();
  await expect(comparisonLab.getByRole("status")).toHaveText(
    /value model can reduce variance/i,
  );
  await comparisonLab.getByRole("button", { name: /^GRPO$/i }).click();
  await expect(comparisonLab.getByRole("status")).toHaveText(
    /avoiding the value model simplifies/i,
  );

  const lengthLab = page.getByTestId("length-incentive-lab");
  await expect(lengthLab.getByRole("status")).toHaveText(/5\.0x/i);
  await lengthLab
    .getByRole("button", { name: /equalized contribution/i })
    .click();
  await expect(lengthLab.getByRole("status")).toHaveText(/reasoning quality/i);

  const r1Pipeline = page.getByTestId("r1-pipeline-lab");
  await r1Pipeline.getByRole("button", { name: /^R1$/i }).click();
  await expect(r1Pipeline.getByRole("status")).toHaveText(
    /small reasoning sft cold start/i,
  );

  const distillationLab = page.getByTestId("distillation-lab");
  await distillationLab
    .getByRole("button", { name: /ordinary distillation/i })
    .click();
  await expect(distillationLab.getByRole("status")).toHaveText(
    /distributional knowledge/i,
  );

  const misconceptionCheck = page.getByTestId("reasoning-misconception-check");
  await misconceptionCheck
    .getByRole("button", { name: /always force extended thinking/i })
    .click();
  await expect(misconceptionCheck.getByRole("status")).toHaveText(/not quite/i);
  await misconceptionCheck
    .getByRole("button", { name: /spend more thinking/i })
    .click();
  await expect(misconceptionCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the standalone Stanford CME295 Lecture 7 page and supports systems labs", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/lecture-7");

  await expect(
    page.getByRole("heading", {
      name: /connect language models to systems they can inspect and act through/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /route the request before choosing the system/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /retrieval is a recall stage before a precision stage/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /tool calls split prediction from execution/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /an agent is a loop, not a single tool call/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toHaveCount(0);
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*NDCG/)).toHaveCount(0);

  const routerLab = page.getByTestId("request-router-lab");
  await routerLab.getByRole("button", { name: /exact record/i }).click();
  await routerLab.getByRole("button", { name: /tool call/i }).click();
  await expect(routerLab.getByRole("status")).toHaveText(/correct route/i);

  const retrievalWorkbench = page.getByTestId("retrieval-workbench");
  await retrievalWorkbench.getByRole("button", { name: /^Hybrid$/i }).click();
  await retrievalWorkbench
    .getByRole("button", { name: /cross-encoder rerank/i })
    .click();
  await expect(retrievalWorkbench.getByRole("status")).toHaveText(/NDCG@5/i);

  const toolConsole = page.getByTestId("tool-calling-console");
  await toolConsole.getByRole("button", { name: /calculator/i }).click();
  await expect(toolConsole.getByRole("status")).toHaveText(/calculate/i);

  const agentLoop = page.getByTestId("agent-loop-lab");
  await agentLoop.getByRole("button", { name: /advance loop/i }).click();
  await expect(agentLoop.getByRole("status")).toHaveText(
    /current temperature/i,
  );

  const safetyLab = page.getByTestId("safety-lab");
  await safetyLab.getByRole("button", { name: /human approval/i }).click();
  await safetyLab.getByRole("button", { name: /egress filter/i }).click();
  await expect(safetyLab.getByRole("status")).toHaveText(/0 \/ 10/i);
});

test("renders the standalone Stanford CME295 Lecture 8 page and supports evaluation labs", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/lecture-8");

  await expect(
    page.getByRole("heading", {
      name: /build the evaluation console before improving the model/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /first separate answer quality from system behavior/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /raw agreement needs a chance baseline/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /reference overlap is useful and brittle/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /localize the failure before choosing a fix/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toHaveCount(0);
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*kappa/)).toHaveCount(0);

  const scopeRouter = page.getByTestId("evaluation-scope-router");
  await scopeRouter.getByRole("button", { name: /refund agent/i }).click();
  await expect(scopeRouter.getByRole("status")).toHaveText(
    /inspect the workflow/i,
  );

  const agreementLab = page.getByTestId("human-agreement-lab");
  await agreementLab.getByRole("button", { name: /lenient raters/i }).click();
  await expect(agreementLab.getByRole("status")).toHaveText(/0\.80 \* 0\.70/i);

  const overlapLab = page.getByTestId("reference-overlap-lab");
  await overlapLab
    .getByRole("button", { name: /meaningful paraphrase/i })
    .click();
  await expect(overlapLab.getByRole("status")).toHaveText(/precision/i);

  const judgeLab = page.getByTestId("judge-bias-lab");
  await judgeLab.getByRole("button", { name: /swap and aggregate/i }).click();
  await expect(judgeLab.getByRole("status")).toHaveText(/position bias/i);

  const factualityLab = page.getByTestId("factuality-claim-lab");
  await factualityLab.getByRole("button", { name: /half credit/i }).click();
  await expect(factualityLab.getByRole("status")).toHaveText(/65%/i);

  const agentLab = page.getByTestId("agent-failure-lab");
  await agentLab.getByRole("button", { name: /wrong argument/i }).click();
  await expect(agentLab.getByRole("status")).toHaveText(/helper tool/i);

  const benchmarkLab = page.getByTestId("benchmark-reliability-lab");
  await benchmarkLab.getByRole("button", { name: /k=3/i }).click();
  await expect(benchmarkLab.getByRole("status")).toHaveText(
    /Pass@3 = 99% but Pass\^3 = 51%/i,
  );
});

test("renders the Stanford CME295 Lecture 9 synthesis page and supports recap labs", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect9");

  await expect(
    page.getByRole("heading", {
      name: /reconstruct the transformer course from the recap/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /rebuild lectures 1-8 as mechanisms/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /same transformer vocabulary, new input units/i,
    }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*Attention/)).toHaveCount(0);
  await expect(page.getByText(/course recap\s*70%/i)).toHaveCount(0);
  await expect(page.getByText(/30 min plan/i)).toHaveCount(0);
  await expect(page.getByText(/readiness:/i)).toHaveCount(0);

  const recapAtlas = page.getByTestId("lecture9-recap-atlas");
  await expect(
    recapAtlas.locator("[data-testid^='lecture9-recap-unit-']"),
  ).toHaveCount(9);
  await expect(
    page.getByTestId("lecture9-recap-unit-representation"),
  ).toContainText(/sequence of token ids/i);
  await expect(page.getByTestId("lecture9-recap-unit-attention")).toContainText(
    /Query/i,
  );
  await expect(
    recapAtlas.getByText(/GRPO removes the separate value model/i),
  ).toBeVisible();

  const courseTrace = page.getByTestId("lecture9-course-trace");
  await courseTrace
    .getByRole("button", { name: /external state added/i })
    .click();
  await expect(courseTrace.getByRole("status")).toHaveText(
    /RAG, tools, and agents add external state/i,
  );

  const transferLab = page.getByTestId("vision-diffusion-transfer-lab");
  await transferLab.getByRole("button", { name: /vit patches/i }).click();
  await expect(transferLab.getByRole("status")).toHaveText(
    /Patch means fixed image square/i,
  );
  await transferLab.getByRole("button", { name: /masked diffusion/i }).click();
  await transferLab.getByRole("button", { name: /8 tokens per pass/i }).click();
  await expect(transferLab.getByRole("status")).toHaveText(/3 passes/i);
  await expect(transferLab.getByRole("status")).toHaveText(/8\.0x/i);
  await transferLab.getByRole("button", { name: /vlm wiring/i }).click();
  await transferLab
    .getByRole("button", { name: /cross-attend to image memory/i })
    .click();
  await expect(transferLab.getByRole("status")).toHaveText(
    /separate set of visual encoder features/i,
  );

  const layerCheck = page.getByTestId("lecture9-layer-check");
  await layerCheck
    .getByRole("button", { name: /only retrain the base model/i })
    .click();
  await expect(layerCheck.getByRole("status")).toHaveText(/Not yet/i);
  await layerCheck
    .getByRole("button", { name: /add retrieval or a policy tool/i })
    .click();
  await expect(layerCheck.getByRole("status")).toHaveText(/Correct/i);
});

test("renders the Probability L3 learning page and supports checks", async ({
  page,
}) => {
  const selectionPermissionErrors = trackSelectionPermissionErrors(page);

  await page.goto("/learn/crash-course-probability/crash-probability-l3");

  await expect(
    page.getByRole("heading", {
      name: /make the observed answer more probable/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /softmax explorer/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\$\$P\(y_i\)/)).toHaveCount(0);

  const logitCheck = page.getByTestId("logit-check");
  await logitCheck
    .getByRole("button", { name: /already probabilities/i })
    .click();
  await expect(logitCheck.getByRole("status")).toHaveText(/not yet/i);

  await logitCheck.getByRole("button", { name: /highest raw score/i }).click();
  await expect(logitCheck.getByRole("status")).toHaveText(/correct/i);

  const heroSummary = page.getByText(/neural-network training looks less/i);
  const box = await heroSummary.boundingBox();
  expect(box, "learning text should be visible").not.toBeNull();
  if (box) {
    await page.mouse.move(box.x + 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(
      box.x + Math.min(box.width - 2, 360),
      box.y + box.height / 2,
      { steps: 12 },
    );
    await page.mouse.up();
  }

  expect(
    selectionPermissionErrors,
    "selecting learning text should not produce React permission errors",
  ).toEqual([]);
});

test("renders the Probability L4 learning page and supports gridworld checks", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-probability/crash-probability-l4");

  await expect(
    page.getByRole("heading", {
      name: /choose actions by averaging possible futures/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /gridworld expected-return lab/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /epsilon-greedy policy lab/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\$\$P\(s'/)).toHaveCount(0);

  const gridworldLab = page.getByTestId("gridworld-decision-lab");
  await gridworldLab.getByRole("button", { name: /^Up$/i }).click();
  await expect(page.getByTestId("gridworld-summary")).toHaveText(
    /selected action: up/i,
  );

  const markovCheck = page.getByTestId("markov-check");
  await markovCheck
    .getByRole("button", { name: /latest message is the current observation/i })
    .click();
  await expect(markovCheck.getByRole("status")).toHaveText(/not yet/i);

  await markovCheck
    .getByRole("button", { name: /includes relevant history or memory/i })
    .click();
  await expect(markovCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the Probability L5 learning page and supports generation labs", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-probability/crash-probability-l5");

  await expect(
    page.getByRole("heading", {
      name: /turn uncertainty into generated output/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /sampling distribution lab/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /generation is a distribution, a rule, then a draw/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /what each control means/i }),
  ).toBeVisible();
  await expect(page.getByText(/what the lab numbers mean/i)).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /latent house probe/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /marginalize over hidden alternatives/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /forward noise first, learned reverse process second/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /diffusion denoising path/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\$\$x\\sim/)).toHaveCount(0);

  const samplingLab = page.getByTestId("sampling-lab");
  await samplingLab.getByRole("button", { name: /^Greedy$/i }).click();
  await expect(page.getByTestId("sampling-summary")).toHaveText(
    /greedy selects mat/i,
  );
  await samplingLab.getByRole("button", { name: /draw next sample/i }).click();
  await samplingLab.getByRole("button", { name: /^Top-k$/i }).click();
  await expect(page.getByTestId("sampling-summary")).toHaveText(
    /top-k selects mat/i,
  );

  const temperatureCheck = page.getByTestId("temperature-knowledge-check");
  await temperatureCheck
    .getByRole("button", { name: /adds new knowledge/i })
    .click();
  await expect(temperatureCheck.getByRole("status")).toHaveText(/not yet/i);
  await temperatureCheck
    .getByRole("button", { name: /flattens the distribution/i })
    .click();
  await expect(temperatureCheck.getByRole("status")).toHaveText(/correct/i);

  const latentProbe = page.getByTestId("latent-variable-probe");
  await latentProbe
    .getByLabel(/probe style latent factor/i)
    .selectOption("modern");
  await expect(latentProbe.getByText(/style: Modern/i)).toBeVisible();
  await expect(latentProbe.getByText(/flat roof/i)).toBeVisible();

  const diffusionLab = page.getByTestId("diffusion-path-lab");
  await diffusionLab.getByRole("button", { name: /Seed B/i }).click();
  await diffusionLab.getByRole("button", { name: /Denoise one step/i }).click();
  await expect(page.getByTestId("diffusion-summary")).toHaveText(
    /Seed B \/ t=4/i,
  );
});

test("keeps the learning page usable at mobile width", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/crash-course-probability/crash-probability-l3");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Stanford CME295 Lecture 1 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/cme295-lect1");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Stanford CME295 Lecture 2 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/cme295-lect2");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Stanford CME295 Lecture 3 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/cme295-lect3");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Stanford CME295 Lecture 4 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/cme295-lect4");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Stanford CME295 Lecture 5 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/cme295-lect5");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the standalone Stanford CME295 Lecture 6 page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/lecture-6");

  await expect(
    page.getByRole("link", { name: /back to stanford cme295 course/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the standalone Stanford CME295 Lecture 7 page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/lecture-7");

  await expect(
    page.getByRole("link", { name: /back to stanford cme295 course/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the standalone Stanford CME295 Lecture 8 page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/lecture-8");

  await expect(
    page.getByRole("link", { name: /back to stanford cme295 course/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Stanford CME295 Lecture 9 synthesis page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/cme295-lect9");

  await expect(
    page.getByRole("link", { name: /start synthesis questions/i }).first(),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Probability L4 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/crash-course-probability/crash-probability-l4");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Probability L5 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/crash-course-probability/crash-probability-l5");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("renders the Clinical Trials L3 learning page and supports checks", async ({
  page,
}) => {
  await page.goto("/learn/clinical-trials/clinical-trials-l3");

  await expect(
    page.getByRole("heading", {
      name: /read the size, uncertainty, and clinical meaning/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /treatment effect explorer/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[\\text\{ARR\}/)).toHaveCount(0);

  const absoluteRiskCheck = page.getByTestId("absolute-risk-check");
  await absoluteRiskCheck
    .getByRole("button", { name: /always clinically large/i })
    .click();
  await expect(absoluteRiskCheck.getByRole("status")).toHaveText(/not yet/i);

  await absoluteRiskCheck
    .getByRole("button", { name: /absolute benefit/i })
    .click();
  await expect(absoluteRiskCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the AI Agents memory survey learning page and supports the design workbench", async ({
  page,
}) => {
  await page.goto("/learn/ai-agents/ai-agents-memory-survey");

  await expect(
    page.getByRole("heading", {
      name: /design memory as a managed subsystem/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /build the memory system the scenario actually needs/i,
    }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*M_\{t\+1\}/)).toHaveCount(0);

  const boundaryLab = page.getByTestId("memory-boundary-lab");
  await boundaryLab.getByRole("button", { name: /classical rag/i }).click();
  await expect(
    boundaryLab.getByText(/retrieve external knowledge/i),
  ).toBeVisible();

  const workbench = page.getByTestId("memory-design-workbench");
  await expect(workbench.getByRole("status")).toHaveText(
    /strong survey-aligned design/i,
  );
  await workbench.getByRole("button", { name: /base weights/i }).click();
  await expect(workbench.getByRole("status")).toHaveText(/hard to audit/i);
  await workbench.getByRole("button", { name: /coding agent/i }).click();
  await workbench.getByRole("button", { name: /^skills/i }).click();
  await workbench.getByRole("button", { name: /adapters/i }).click();
  await expect(workbench.getByText(/modular competence/i)).toBeVisible();

  const frontierBoard = page.getByTestId("memory-frontier-board");
  await frontierBoard.getByRole("button", { name: /shared memory/i }).click();
  await expect(frontierBoard.getByText(/common substrate/i)).toBeVisible();
});

test("renders the AI Agents agent-native memory learning page and supports the evaluation console", async ({
  page,
}) => {
  await page.goto("/learn/ai-agents/ai-agents-agent-native-memory");

  await expect(
    page.getByRole("heading", {
      name: /evaluate memory like infrastructure/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /stress a memory architecture before trusting the answer/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /retrieval is evidence assembly/i,
    }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*M_\{sys\}/)).toHaveCount(0);

  const moduleRack = page.getByTestId("agent-native-module-rack");
  await moduleRack
    .getByRole("button", { name: /Q: Retrieval and routing/i })
    .click();
  await expect(
    moduleRack.getByText(/How does a query find useful evidence/i),
  ).toBeVisible();

  const console = page.getByTestId("agent-native-evaluation-console");
  await expect(console.getByRole("status")).toHaveText(/strong workload fit/i);
  await console
    .getByRole("button", { name: /Latest-state fact update/i })
    .click();
  await console
    .getByRole("button", { name: /Relation-aware graph memory/i })
    .click();
  await expect(console.getByText(/Temporal update fidelity/i)).toBeVisible();
  await console.getByRole("button", { name: /Abstractive summary/i }).click();
  await expect(console.getByText(/summary-heavy/i)).toBeVisible();

  const evidenceLab = page.getByTestId("agent-native-evidence-distance-lab");
  await evidenceLab
    .getByRole("button", { name: /Flat semantic cache/i })
    .click();
  await evidenceLab
    .getByRole("button", { name: /Direct top-1 retrieval/i })
    .click();
  await expect(
    evidenceLab.getByText(/long-range reconstruction/i),
  ).toBeVisible();

  const ablationBoard = page.getByTestId("agent-native-ablation-board");
  await ablationBoard.getByRole("button", { name: /Retrieval/i }).click();
  await expect(ablationBoard.getByText(/Planning only/i)).toBeVisible();
});

test("renders the AI Agents AtomMem learning page and supports the pipeline debugger", async ({
  page,
}) => {
  await page.goto("/learn/ai-agents/ai-agents-atommem");

  await expect(
    page.getByRole("heading", {
      name: /debug a memory pipeline before trusting its answer/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /run one memory design and watch where it fails/i,
    }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*S_h/)).toHaveCount(0);

  const representationLab = page.getByTestId("atommem-representation-lab");
  await representationLab.getByRole("button", { name: /raw log/i }).click();
  await expect(representationLab.getByRole("status")).toHaveText(
    /noise-heavy retrieval/i,
  );

  const debuggerLab = page.getByTestId("atommem-pipeline-debugger");
  await expect(debuggerLab.getByRole("status")).toHaveText(
    /strong atommem-style design/i,
  );
  await debuggerLab
    .getByRole("button", { name: /graph recall enabled/i })
    .click();
  await expect(
    debuggerLab.getByText(/graph recall is disabled/i),
  ).toBeVisible();
  await debuggerLab.getByRole("button", { name: /^0\.9$/i }).click();
  await expect(debuggerLab.getByText(/high event weight/i)).toBeVisible();

  const ablationBoard = page.getByTestId("atommem-ablation-board");
  await ablationBoard.getByRole("button", { name: /w\/o graph/i }).click();
  await expect(ablationBoard.getByText(/remote dependencies/i)).toBeVisible();
});

test("renders the standalone Medicine Lecture 1 page and supports clinical reasoning interactions", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-medicine/lecture-1");

  await expect(
    page.getByRole("heading", {
      name: /what medicine is/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /work through a patient case in clinical order/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toHaveCount(0);

  const studio = page.getByTestId("medicine-clinical-studio");
  await studio.getByRole("button", { name: /exam and vitals/i }).click();
  await expect(
    studio.getByRole("button", { name: /^Oxygen saturation 89%$/i }),
  ).toBeVisible();

  await studio.getByRole("button", { name: /^model$/i }).click();
  await expect(studio.locator(".representation-output")).toContainText(
    /hypoxemia/i,
  );

  await studio.getByRole("button", { name: /give oxygen/i }).click();
  await expect(
    studio.getByText(/low oxygen is a direct instability signal/i),
  ).toBeVisible();

  await studio.getByRole("button", { name: /wait for every test/i }).click();
  await expect(studio.getByText(/premature closure is active/i)).toBeVisible();
});

test("keeps the Clinical Trials L3 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/clinical-trials/clinical-trials-l3");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the AI Agents memory survey learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/ai-agents/ai-agents-memory-survey");

  await expect(
    page.getByRole("link", { name: /start memory survey questions/i }).first(),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the AI Agents agent-native memory learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/ai-agents/ai-agents-agent-native-memory");

  await expect(
    page
      .getByRole("link", { name: /start agent-native memory questions/i })
      .first(),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the AI Agents AtomMem learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/ai-agents/ai-agents-atommem");

  await expect(
    page.getByRole("link", { name: /start atommem questions/i }).first(),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the standalone Medicine Lecture 1 page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/crash-course-medicine/lecture-1");

  await expect(
    page.getByRole("link", { name: /back to medicine course/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("transitions from Stanford CME295 Lecture 1 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect1");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect1$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 1: transformers & llms/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 69/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Stanford CME295 Lecture 2 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect2");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect2$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 2: transformer-based models & tricks/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 85/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Stanford CME295 Lecture 3 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect3");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect3$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 3: large language models, moe & inference/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 80/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Stanford CME295 Lecture 4 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect4");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect4$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 4: llm training, scaling & alignment/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 80/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Stanford CME295 Lecture 5 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect5");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect5$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 5: llm preference tuning, rlhf & dpo/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Stanford CME295 reasoning learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/lecture-6");

  await page.getByRole("link", { name: /start reasoning questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect6$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 6: llm reasoning & test-time scaling/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Stanford CME295 evaluation learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/lecture-8");

  await page.getByRole("link", { name: /start evaluation questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect8$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 8: llm evaluation/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Stanford CME295 synthesis learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect9");

  await page
    .getByRole("link", { name: /start synthesis questions/i })
    .first()
    .click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect9$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 9: course synthesis & frontiers/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from learning into the matching quiz source", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-probability/crash-probability-l3");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=crash-probability-l3$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /crash course probability l3: likelihood, loss, softmax, and deep learning/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Probability L4 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-probability/crash-probability-l4");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=crash-probability-l4$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /crash course probability l4: probability over time: reinforcement learning/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Probability L5 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-probability/crash-probability-l5");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=crash-probability-l5$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /crash course probability l5: sampling, latent variables, and diffusion models/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Clinical Trials L3 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/clinical-trials/clinical-trials-l3");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=clinical-trials-l3$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /clinical trials crash course l3: statistics and evidence interpretation/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from AI Agents memory survey learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/ai-agents/ai-agents-memory-survey");

  await page
    .getByRole("link", { name: /start memory survey questions/i })
    .first()
    .click();

  await expect(page).toHaveURL(/\/\?source=ai-agents-memory-survey$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /ai agents: memory in the age of ai agents/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 40/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from AI Agents agent-native memory learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/ai-agents/ai-agents-agent-native-memory");

  await page
    .getByRole("link", { name: /start agent-native memory questions/i })
    .first()
    .click();

  await expect(page).toHaveURL(/\/\?source=ai-agents-agent-native-memory$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /ai agents: are we ready for an agent-native memory system\?/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 40/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from AI Agents AtomMem learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/ai-agents/ai-agents-atommem");

  await page
    .getByRole("link", { name: /start atommem questions/i })
    .first()
    .click();

  await expect(page).toHaveURL(/\/\?source=ai-agents-atommem$/, {
    timeout: ROUTE_TRANSITION_TIMEOUT_MS,
  });
  await expect(
    page.getByRole("heading", {
      name: /ai agents: atommem/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 40/i)).toBeVisible({
    timeout: 10000,
  });
});
