import { expect, test, type Page } from "@playwright/test";

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
  );
});

test("labels and sorts course learning experiences by source sequence", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295");

  const courseCards = page.locator(
    'section[aria-label="Stanford CME295 Transformers & LLMs learning experiences"] a',
  );
  await expect(courseCards).toHaveCount(4);

  const cardTexts = await courseCards.allTextContents();
  expect(cardTexts[0]).toContain("Lecture 1");
  expect(cardTexts[1]).toContain("Lecture 2");
  expect(cardTexts[2]).toContain("Lecture 3");
  expect(cardTexts[3]).toContain("Lecture 4");
  expect(cardTexts.join("\n")).not.toMatch(/15 min|16 min|18 min|19 min/i);
  expect(cardTexts.join("\n")).not.toMatch(
    /Introductory NLP with ML basics|After CME295 Lecture/i,
  );
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

test("transitions from Stanford CME295 Lecture 1 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect1");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect1$/);
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

  await expect(page).toHaveURL(/\/\?source=cme295-lect2$/);
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

  await expect(page).toHaveURL(/\/\?source=cme295-lect3$/);
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

  await expect(page).toHaveURL(/\/\?source=cme295-lect4$/);
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

test("transitions from learning into the matching quiz source", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-probability/crash-probability-l3");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=crash-probability-l3$/);
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

  await expect(page).toHaveURL(/\/\?source=crash-probability-l4$/);
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

  await expect(page).toHaveURL(/\/\?source=crash-probability-l5$/);
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

  await expect(page).toHaveURL(/\/\?source=clinical-trials-l3$/);
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
