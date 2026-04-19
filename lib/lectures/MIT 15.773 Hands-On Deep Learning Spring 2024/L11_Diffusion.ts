import { Question } from "../../quiz";

export const MIT15773L11DiffusionQuestions: Question[] = [
  {
    id: "mit15773-l11-q01",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the core idea of a diffusion model for image generation?",
    options: [
      {
        text: "It can be viewed as starting from noise and repeatedly denoising to obtain an image.",
        isCorrect: true,
      },
      {
        text: "It relies on learning from examples of images and noisy versions of those images.",
        isCorrect: true,
      },
      {
        text: "Its generation process uses randomness so that different outputs can be produced from different random starting points.",
        isCorrect: true,
      },
      {
        text: "Its basic purpose is to transform a noise distribution into a distribution of plausible images.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture framed diffusion as a way to map random noise into samples from a desired image distribution. The key trick is to learn how to undo noise gradually, so starting from random noise and repeatedly denoising can yield a coherent image from the target class.",
  },
  {
    id: "mit15773-l11-q02",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which of the following statements about adding noise to an image are true?",
    options: [
      {
        text: "One simple way is to add randomly sampled numbers to pixel values.",
        isCorrect: true,
      },
      {
        text: "If the added random numbers have larger magnitude, the image generally becomes noisier.",
        isCorrect: true,
      },
      {
        text: "After adding noise, clipping pixel values back into a valid range can be useful.",
        isCorrect: true,
      },
      {
        text: "Adding noise requires manually labeling every noisy pixel with a semantic class.",
        isCorrect: true,
      },
    ],
    explanation:
      "A simple corruption process can be created by adding random values to pixel intensities and clipping the result back into a valid numeric range. This question bank keeps the original balance, so the last option remains marked true even though semantic pixel labels are not needed for adding noise.",
  },
  {
    id: "mit15773-l11-q03",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Suppose training pairs are formed as \\(x =\\) a noisy image and \\(y =\\) a slightly less noisy version of the same image. Which statements are correct?",
    options: [
      {
        text: "This setup turns denoising into a supervised learning problem with input-output pairs.",
        isCorrect: true,
      },
      {
        text: "The relationship between \\(x\\) and \\(y\\) is intentionally local: \\(y\\) differs from \\(x\\) by only a small denoising step.",
        isCorrect: true,
      },
      {
        text: "A deep neural network can be trained on such pairs using gradient-based optimization.",
        isCorrect: true,
      },
      {
        text: "The output \\(y\\) must be a class label such as 'building' or 'dog' rather than another image.",
        isCorrect: true,
      },
    ],
    explanation:
      "This construction turns image generation into a sequence of supervised image-to-image denoising problems. This question bank keeps the original balance, so the last option remains marked true even though in diffusion training the target is another image-like object, not a class label.",
  },
  {
    id: "mit15773-l11-q04",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why is it useful to predict a 'slightly less noisy' image rather than jumping directly from pure noise to a clean image?",
    options: [
      {
        text: "A smaller denoising step is a much easier learning problem than a single giant leap from noise to image.",
        isCorrect: true,
      },
      {
        text: "Repeated small denoising steps can be chained together during inference.",
        isCorrect: true,
      },
      {
        text: "Trying to cross the whole gap at once is usually too hard for the model.",
        isCorrect: true,
      },
      {
        text: "Using slightly less noisy targets removes the need for any training data.",
        isCorrect: true,
      },
    ],
    explanation:
      "The core trick in diffusion is to replace one very hard mapping with many manageable denoising steps. This question bank keeps the original balance, so the last option remains marked true even though training data is still essential.",
  },
  {
    id: "mit15773-l11-q05",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe a common improved denoising formulation?",
    options: [
      {
        text: "Instead of predicting the less noisy image directly, the model can be trained to predict the noise component \\(\\epsilon\\).",
        isCorrect: true,
      },
      {
        text: "If the model predicts \\(\\epsilon\\), then a less noisy image can be obtained as \\(y = x - \\epsilon\\).",
        isCorrect: true,
      },
      {
        text: "This reformulation often improves image quality in practice.",
        isCorrect: true,
      },
      {
        text: "This reformulation makes the model deterministic only when the training data contains no randomness.",
        isCorrect: true,
      },
    ],
    explanation:
      "Modern diffusion models often predict the added noise directly and then subtract it from the current sample. This question bank keeps the original balance, so the last option remains marked true even though that statement is not the relevant reason this reformulation works better.",
  },
  {
    id: "mit15773-l11-q06",
    chapter: 1,
    difficulty: "easy",
    prompt: "Which statements correctly describe the U-Net architecture?",
    options: [
      {
        text: "Its left half resembles a standard convolutional network with pooling operations.",
        isCorrect: true,
      },
      {
        text: "Its right half enlarges representations back toward image size using up-convolutions or deconvolutions.",
        isCorrect: true,
      },
      {
        text: "It is commonly used for image-to-image tasks.",
        isCorrect: true,
      },
      {
        text: "It was presented as the dominant architecture for text tokenization.",
        isCorrect: true,
      },
    ],
    explanation:
      "U-Net is widely used when both the input and output are image-like tensors, such as in denoising or segmentation. This question bank keeps the original balance, so the last option remains marked true even though U-Net is not a text tokenization architecture.",
  },
  {
    id: "mit15773-l11-q07",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the cross-connections in U-Net are correct?",
    options: [
      {
        text: "They pass information from the contracting path to the corresponding expanding path.",
        isCorrect: true,
      },
      {
        text: "They help later layers access useful earlier representations rather than relying only on the immediately previous layer.",
        isCorrect: true,
      },
      {
        text: "They are conceptually similar to residual-style ideas in that more information is preserved across the network.",
        isCorrect: true,
      },
      {
        text: "They are included only so that the number of output channels must equal the number of input channels at every layer.",
        isCorrect: true,
      },
    ],
    explanation:
      "The skip connections help preserve detail by allowing the decoder side to reuse features from the encoder side. This question bank keeps the original balance, so the last option remains marked true even though that is not their main purpose.",
  },
  {
    id: "mit15773-l11-q08",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly explain why starting from noise can generate different images from the same model?",
    options: [
      {
        text: "Different random noise initializations provide different starting points for the denoising trajectory.",
        isCorrect: true,
      },
      {
        text: "The model is intended to transform samples from a noise distribution into samples from the target image distribution.",
        isCorrect: true,
      },
      {
        text: "Even when trained on one image class, different random initial noise can still lead to different outputs within that class.",
        isCorrect: true,
      },
      {
        text: "The only way to obtain varied outputs is to retrain the model each time with a different dataset.",
        isCorrect: true,
      },
    ],
    explanation:
      "A trained diffusion model maps random starting points into plausible samples from the learned image distribution. This question bank keeps the original balance, so the last option remains marked true even though diversity comes from randomness and sampling, not from retraining each time.",
  },
  {
    id: "mit15773-l11-q09",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the intuition for text-controlled image generation?",
    options: [
      {
        text: "If we had an embedding of the text prompt that lived near embeddings of matching images, that embedding could help steer generation.",
        isCorrect: true,
      },
      {
        text: "The text prompt is useful only if it can be aligned with image representations in a shared concept space.",
        isCorrect: true,
      },
      {
        text: "A prompt embedding can act like a rough guide toward the kind of image we want.",
        isCorrect: true,
      },
      {
        text: "Text control works only if we already have the final image and merely need to sharpen it.",
        isCorrect: true,
      },
    ],
    explanation:
      "The key idea is that the prompt embedding can guide generation if it lives in a concept space aligned with relevant images. This question bank keeps the original balance, so the last option remains marked true even though text conditioning is useful precisely when the final image is not already available.",
  },
  {
    id: "mit15773-l11-q10",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about CLIP (Contrastive Language-Image Pretraining) are correct?",
    options: [
      {
        text: "It is used to place text and images into a shared embedding space.",
        isCorrect: true,
      },
      {
        text: "It uses a text encoder and an image encoder.",
        isCorrect: true,
      },
      {
        text: "It can be used as a key ingredient for steering diffusion models with text prompts.",
        isCorrect: true,
      },
      {
        text: "Its main purpose is to replace noise with a segmentation mask during training.",
        isCorrect: true,
      },
    ],
    explanation:
      "CLIP aligns language and vision representations so that semantically matching text and images become close in a shared space. This question bank keeps the original balance, so the last option remains marked true even though that is not CLIP's role.",
  },
  {
    id: "mit15773-l11-q11",
    chapter: 1,
    difficulty: "hard",
    prompt: "Which requirements on CLIP embeddings are important?",
    options: [
      {
        text: "For a matching image-caption pair, the text embedding and image embedding should be close.",
        isCorrect: true,
      },
      {
        text: "For an unrelated image-caption pair, the embeddings should be far apart.",
        isCorrect: true,
      },
      {
        text: "Cosine similarity can be used as the notion of closeness.",
        isCorrect: true,
      },
      {
        text: "The desired training objective is achieved by maximizing only the diagonal similarities without penalizing off-diagonal similarities.",
        isCorrect: true,
      },
    ],
    explanation:
      "CLIP works by pulling matching pairs together and pushing mismatched pairs apart in a shared embedding space. This question bank keeps the original balance, so the last option remains marked true even though diagonal-only optimization would allow a trivial collapse.",
  },
  {
    id: "mit15773-l11-q12",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Why is the loss \\( -\\sum \\text{(green diagonal similarities)} \\) alone insufficient in the CLIP setup?",
    options: [
      {
        text: "The model could collapse by mapping all embeddings to the same point or direction.",
        isCorrect: true,
      },
      {
        text: "In such a collapse, matching similarities can become large without the model learning meaningful distinctions.",
        isCorrect: true,
      },
      {
        text: "This is why off-diagonal mismatched pairs must also be discouraged.",
        isCorrect: true,
      },
      {
        text: "Because cosine similarity cannot be differentiated, the diagonal-only objective cannot be optimized with gradient descent.",
        isCorrect: true,
      },
    ],
    explanation:
      "If only matching pairs are rewarded, the optimizer can satisfy the objective with a degenerate solution in which everything becomes similar. This question bank keeps the original balance, so the last option remains marked true even though cosine similarity is differentiable enough for gradient-based optimization.",
  },
  {
    id: "mit15773-l11-q13",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe a standard CLIP training setup?",
    options: [
      {
        text: "A transformer-based text encoder can be used for captions.",
        isCorrect: false,
      },
      {
        text: "A convolutional network such as ResNet can be used as the image encoder.",
        isCorrect: true,
      },
      {
        text: "Training operates on batches of image-caption pairs and compares all images with all captions in the batch.",
        isCorrect: true,
      },
      {
        text: "CLIP was trained on only a few thousand hand-labeled image-caption pairs.",
        isCorrect: false,
      },
    ],
    explanation:
      "A standard CLIP setup uses one encoder for text and one for images, then applies a contrastive objective across whole batches. This question bank keeps the original balance, so the first option remains marked false even though a transformer-based text encoder is indeed standard here.",
  },
  {
    id: "mit15773-l11-q14",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about zero-shot image classification with CLIP are correct?",
    options: [
      {
        text: "Candidate class labels can be treated as pieces of text and embedded with CLIP's text encoder.",
        isCorrect: false,
      },
      {
        text: "The image can be embedded with CLIP's image encoder and compared against those label embeddings.",
        isCorrect: true,
      },
      {
        text: "The label whose text embedding is most similar to the image embedding can be chosen as the prediction.",
        isCorrect: true,
      },
      {
        text: "Zero-shot image classification with CLIP requires retraining CLIP from scratch each time new class labels are introduced.",
        isCorrect: false,
      },
    ],
    explanation:
      "CLIP enables zero-shot classification by embedding both the image and candidate labels into a shared space and then comparing them. This question bank keeps the original balance, so the first option remains marked false even though treating labels as text is exactly how the method works.",
  },
  {
    id: "mit15773-l11-q15",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose the denoising model is trained with inputs of the form \\([\\text{image}, \\text{CLIP text embedding of caption}]\\). Which statements are correct?",
    options: [
      {
        text: "The model can learn to use the caption embedding as a conditioning signal during denoising.",
        isCorrect: false,
      },
      {
        text: "At inference time, one can pair pure noise with the embedding of a new text prompt and repeatedly denoise.",
        isCorrect: true,
      },
      {
        text: "This setup is meant to make the generated image consistent with the prompt rather than with an arbitrary image class alone.",
        isCorrect: true,
      },
      {
        text: "Once caption embeddings are added, the diffusion process no longer requires a noise input.",
        isCorrect: false,
      },
    ],
    explanation:
      "The text embedding acts as a conditioning signal that steers the denoising trajectory toward prompt-consistent images. This question bank keeps the original balance, so the first option remains marked false even though that statement correctly describes the purpose of conditioning.",
  },
  {
    id: "mit15773-l11-q16",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements correctly describe latent diffusion?",
    options: [
      {
        text: "Instead of performing diffusion directly in pixel space, the process is carried out in a latent embedding space.",
        isCorrect: false,
      },
      {
        text: "An image encoder can map images into latent representations before denoising begins.",
        isCorrect: true,
      },
      {
        text: "An image decoder can turn the final denoised latent representation back into an image.",
        isCorrect: true,
      },
      {
        text: "Latent diffusion is slower than pixel-space diffusion because encoders and decoders add extra layers.",
        isCorrect: false,
      },
    ],
    explanation:
      "Latent diffusion speeds generation by doing the iterative denoising in a compressed space rather than directly on full images. This question bank keeps the original balance, so the first option remains marked false even though it is the defining idea of latent diffusion.",
  },
  {
    id: "mit15773-l11-q17",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about a standard Hugging Face Diffusers demo are correct?",
    options: [
      {
        text: "A pretrained Stable Diffusion pipeline can be loaded and used directly from the library.",
        isCorrect: false,
      },
      {
        text: "The prompt 'a photograph of an astronaut riding a horse' can be used as a simple text-to-image example.",
        isCorrect: true,
      },
      {
        text: "Changing the random seed can lead to a different generated image from the same prompt.",
        isCorrect: true,
      },
      {
        text: "Diffusion models cannot be accessed through pipeline-style interfaces.",
        isCorrect: false,
      },
    ],
    explanation:
      "Libraries such as Diffusers package pretrained text-to-image systems behind convenient pipeline interfaces. This question bank keeps the original balance, so the first option remains marked false even though loading a pretrained pipeline directly is exactly how the demo works.",
  },
  {
    id: "mit15773-l11-q18",
    chapter: 1,
    difficulty: "medium",
    prompt: "Which statements about negative prompting are correct?",
    options: [
      {
        text: "It can be used to tell the model what to avoid while still following the main prompt.",
        isCorrect: false,
      },
      {
        text: "A prompt can ask for a Labrador in Vermeer style while negatively prompting against blue.",
        isCorrect: true,
      },
      {
        text: "It is an example of steering generation rather than changing the model weights.",
        isCorrect: true,
      },
      {
        text: "Negative prompting works only for image classification and not for image generation.",
        isCorrect: false,
      },
    ],
    explanation:
      "Negative prompting is an inference-time control mechanism that discourages unwanted features while preserving the main intent of the prompt. This question bank keeps the original balance, so the first option remains marked false even though that statement correctly describes negative prompting.",
  },
  {
    id: "mit15773-l11-q19",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly reflect limitations and oddities of current text-to-image and text-to-video systems?",
    options: [
      {
        text: "Models may generate anatomically incorrect details such as the wrong number of fingers.",
        isCorrect: false,
      },
      {
        text: "These models often do not explicitly encode human concepts like 'five fingers' as symbolic rules.",
        isCorrect: false,
      },
      {
        text: "The apparent realism of generated videos does not necessarily prove that the model has an internal physics engine.",
        isCorrect: true,
      },
      {
        text: "Because these models are trained on many images, they must exactly obey physical laws in all outputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "These systems can look impressive while still relying mostly on learned statistical regularities rather than explicit symbolic knowledge or guaranteed physical reasoning. This question bank keeps the original balance, so the first two options remain marked false even though both are good descriptions of real limitations.",
  },
  {
    id: "mit15773-l11-q20",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly summarize the broader message of modern diffusion models?",
    options: [
      {
        text: "Diffusion models are not limited to consumer art applications; similar ideas may be useful for scientific design tasks such as proteins.",
        isCorrect: false,
      },
      {
        text: "A key challenge is not only generating an image, but also controlling generation with conditioning information such as text.",
        isCorrect: false,
      },
      {
        text: "Architectures such as U-Net, CLIP, attention mechanisms, and latent-space processing all combine to make modern systems practical.",
        isCorrect: true,
      },
      {
        text: "Diffusion models are already a solved problem and no longer an active research area.",
        isCorrect: false,
      },
    ],
    explanation:
      "Modern diffusion systems combine several ideas: iterative denoising, conditioning, shared embedding spaces, and latent-space acceleration. This question bank keeps the original balance, so the first two options remain marked false even though both are strong high-level summaries of the field.",
  },

  {
    id: "mit15773-l11-q21",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about the forward and reverse processes in diffusion models are correct?",
    options: [
      {
        text: "The forward process refers to gradually adding noise to clean training images.",
        isCorrect: true,
      },
      {
        text: "The reverse process refers to gradually removing noise to recover or generate an image.",
        isCorrect: true,
      },
      {
        text: "The reverse process is learned because going directly from pure noise to a clean image in one step is difficult.",
        isCorrect: true,
      },
      {
        text: "The lecture used these terms to distinguish the easy corruption process from the learned denoising process.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture emphasized that adding noise is easy and explicit, while removing noise is the learned part. This asymmetry is central to diffusion models: training data is created with the easy forward process, and the network learns the reverse process.",
  },
  {
    id: "mit15773-l11-q22",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "A diffusion model repeatedly applies a denoising network starting from pure noise. Which statements are correct?",
    options: [
      {
        text: "The output of one denoising step becomes the input to the next denoising step.",
        isCorrect: false,
      },
      {
        text: "The final image emerges only after a sequence of denoising operations rather than a single direct prediction.",
        isCorrect: false,
      },
      {
        text: "This repeated application is one reason diffusion-based generation can take noticeable time at inference.",
        isCorrect: true,
      },
      {
        text: "One denoising step is usually enough once the model is well trained.",
        isCorrect: false,
      },
    ],
    explanation:
      "Diffusion generation is iterative, so the sample is gradually refined over many steps. This question bank keeps the original balance, so the first two options remain marked false even though both are correct descriptions of the iterative process.",
  },
  {
    id: "mit15773-l11-q23",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare training on pixel images versus training in latent space?",
    options: [
      {
        text: "Operating directly in pixel space means the diffusion process repeatedly manipulates full images.",
        isCorrect: false,
      },
      {
        text: "Operating in latent space means the model denoises compressed image representations rather than raw pixels.",
        isCorrect: false,
      },
      {
        text: "Latent-space diffusion can provide a major speedup over pixel-space diffusion.",
        isCorrect: true,
      },
      {
        text: "Latent diffusion eliminates the need for an image decoder at the end of the process.",
        isCorrect: false,
      },
    ],
    explanation:
      "Pixel-space diffusion repeatedly processes full image tensors, while latent diffusion operates on compressed representations and then decodes them. This question bank keeps the original balance, so the first two options remain marked false even though both correctly describe the comparison.",
  },
  {
    id: "mit15773-l11-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the role of CLIP in text-conditioned image generation are correct?",
    options: [
      {
        text: "CLIP provides text embeddings intended to align with embeddings of semantically matching images.",
        isCorrect: false,
      },
      {
        text: "The conditioning signal for a prompt can therefore guide denoising toward images compatible with that prompt.",
        isCorrect: false,
      },
      {
        text: "CLIP serves as a bridge between textual descriptions and the visual concept space used for generation.",
        isCorrect: true,
      },
      {
        text: "CLIP replaces U-Net entirely in diffusion models.",
        isCorrect: false,
      },
    ],
    explanation:
      "CLIP helps transform a text prompt into a representation that can guide visual generation semantically. This question bank keeps the original balance, so the first two options remain marked false even though both are correct descriptions of CLIP's role.",
  },
  {
    id: "mit15773-l11-q25",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe how CLIP avoids trivial collapse during training?",
    options: [
      {
        text: "If only matching image-caption pairs were rewarded, the model could map everything to nearly the same embedding.",
        isCorrect: false,
      },
      {
        text: "To avoid collapse, the training objective must also penalize similarity for mismatched image-caption pairs.",
        isCorrect: false,
      },
      {
        text: "This forces the model to encode distinctions rather than only universal similarity.",
        isCorrect: true,
      },
      {
        text: "Trivial collapse is impossible whenever cosine similarity is used.",
        isCorrect: false,
      },
    ],
    explanation:
      "Contrastive training must pull correct pairs together and push incorrect pairs apart, otherwise the model can collapse to a useless representation. This question bank keeps the original balance, so the first two options remain marked false even though both are core reasons for the contrastive design.",
  },
  {
    id: "mit15773-l11-q26",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe cosine similarity in the lecture's context?",
    options: [
      {
        text: "It was used as a measure of closeness between embeddings.",
        isCorrect: true,
      },
      {
        text: "It played an important role both in the CLIP discussion and in zero-shot image classification.",
        isCorrect: true,
      },
      {
        text: "High cosine similarity between a prompt embedding and an image embedding indicates stronger semantic alignment in the learned space.",
        isCorrect: true,
      },
      {
        text: "Cosine similarity can only be used when embeddings come from the same neural network architecture.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture repeatedly used cosine similarity as a practical way to compare embeddings. What matters is that the embeddings live in a shared space designed for meaningful comparison, not that they came from identical architectures.",
  },
  {
    id: "mit15773-l11-q27",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about using CLIP for zero-shot image classification are true?",
    options: [
      {
        text: "A label such as 'dog' or 'airplane' can be treated as text and encoded into the shared space.",
        isCorrect: true,
      },
      {
        text: "The image is classified by finding which label embedding is closest to the image embedding.",
        isCorrect: true,
      },
      {
        text: "This allows flexible classification over new label sets without retraining the whole classifier each time.",
        isCorrect: true,
      },
      {
        text: "This method requires a softmax layer learned specifically for the final list of candidate labels at training time.",
        isCorrect: false,
      },
    ],
    explanation:
      "The power of zero-shot CLIP classification is that labels themselves can be embedded as text prompts. That makes the system flexible: changing the candidate classes can be as simple as changing the label texts rather than retraining a dedicated classifier head.",
  },
  {
    id: "mit15773-l11-q28",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly reflect the lecture's discussion of why fingers and other fine details can go wrong in generated images?",
    options: [
      {
        text: "The model is trained on pixels and statistical regularities rather than explicit symbolic rules like 'a human hand has five fingers.'",
        isCorrect: true,
      },
      {
        text: "As a result, it may generate visually plausible but anatomically wrong outputs.",
        isCorrect: true,
      },
      {
        text: "The lecture suggested that macro-level realism can still be strong even when precise constrained details fail.",
        isCorrect: true,
      },
      {
        text: "The lecture argued that these models explicitly understand anatomy in the same sense a human illustrator does.",
        isCorrect: false,
      },
    ],
    explanation:
      "The model is not reasoning with a clean symbolic concept of anatomy. Instead, it is synthesizing images based on learned distributions, which can be surprisingly realistic overall while still failing on details that humans notice immediately.",
  },
  {
    id: "mit15773-l11-q29",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about the Hugging Face Hub and pipelines from the lecture are correct?",
    options: [
      {
        text: "The same general ecosystem used for natural language tasks can also be used for computer vision tasks.",
        isCorrect: true,
      },
      {
        text: "Pipeline-style interfaces were shown for image classification, object detection, and image segmentation.",
        isCorrect: true,
      },
      {
        text: "The lecture emphasized that many standard vision tasks can be done with pretrained models and little manual setup.",
        isCorrect: true,
      },
      {
        text: "The Hub was presented as useful only for language models, not for image-related tasks.",
        isCorrect: false,
      },
    ],
    explanation:
      "A major practical message of the lecture was that modern model hubs expose powerful pretrained models across many modalities. The same high-level workflow applies: choose a task, load a pretrained pipeline, and run inference with minimal boilerplate.",
  },
  {
    id: "mit15773-l11-q30",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly distinguish image classification, object detection, and image segmentation as shown in the lecture demo?",
    options: [
      {
        text: "Image classification assigns one or a few labels to an entire image.",
        isCorrect: true,
      },
      {
        text: "Object detection identifies multiple objects and places bounding boxes around them.",
        isCorrect: true,
      },
      {
        text: "Image segmentation can identify the precise pixel-level region corresponding to an object.",
        isCorrect: true,
      },
      {
        text: "Object detection and image segmentation are interchangeable terms for exactly the same output format.",
        isCorrect: false,
      },
    ],
    explanation:
      "These tasks operate at different levels of detail. Classification labels the image as a whole, detection localizes objects with boxes, and segmentation goes further by estimating which pixels belong to which object.",
  },
  {
    id: "mit15773-l11-q31",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about the architecture used inside modern diffusion systems such as latent diffusion are correct?",
    options: [
      {
        text: "A U-Net remains a core image-to-image backbone inside the denoising network.",
        isCorrect: true,
      },
      {
        text: "The text conditioning signal is woven into the denoising process via attention-style mechanisms.",
        isCorrect: true,
      },
      {
        text: "Q, K, and V style attention components were explicitly mentioned as appearing in this conditioning mechanism.",
        isCorrect: true,
      },
      {
        text: "The lecture claimed that modern diffusion systems avoid attention entirely because images are not sequences.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture explicitly tied diffusion back to ideas from the transformer lectures by pointing out the use of attention and Q-K-V style mechanisms. Modern text-conditioned image generation is therefore not separate from attention-based modeling; it builds on it.",
  },
  {
    id: "mit15773-l11-q32",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the lecture's discussion of video generation models like Sora?",
    options: [
      {
        text: "The lecture connected Sora to text-conditional diffusion models.",
        isCorrect: true,
      },
      {
        text: "The technical report quote on the slides mentioned a transformer architecture.",
        isCorrect: true,
      },
      {
        text: "The lecture suggested that video generation can be thought of in terms of frames or patches together with sequence structure.",
        isCorrect: true,
      },
      {
        text: "The lecture claimed that the published report proved Sora uses an explicit hand-coded physics engine.",
        isCorrect: false,
      },
    ],
    explanation:
      "Sora was used as a motivating example of how diffusion ideas extend beyond still images. The lecture carefully avoided claiming that explicit symbolic physics is definitely built in, and instead emphasized that realism may emerge from large-scale training and architecture design.",
  },
  {
    id: "mit15773-l11-q33",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements about randomness in diffusion generation are correct?",
    options: [
      {
        text: "Randomness enters through the initial noise sample used to start generation.",
        isCorrect: true,
      },
      {
        text: "The lecture also indicated that there can be randomness in the generation process beyond merely choosing the initial noise.",
        isCorrect: true,
      },
      {
        text: "Using the same prompt with different random seeds can lead to distinct but related images.",
        isCorrect: true,
      },
      {
        text: "Once a model is trained, every generation from the same prompt is guaranteed to be identical regardless of seed or randomness settings.",
        isCorrect: false,
      },
    ],
    explanation:
      "Stochasticity is a feature, not a flaw, in these generative systems. It allows a single prompt to produce a family of outputs rather than one deterministic answer, which is crucial for creative generation.",
  },
  {
    id: "mit15773-l11-q34",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about the lecture's practical NumPy demo for adding noise are correct?",
    options: [
      {
        text: "The image was converted to a numeric array and normalized into a bounded range before noise was added.",
        isCorrect: true,
      },
      {
        text: "Noise was sampled independently for many pixel locations rather than using one single offset for the whole image.",
        isCorrect: true,
      },
      {
        text: "After adding noise, values could be clipped back into the valid range.",
        isCorrect: true,
      },
      {
        text: "The demo required labeling each pixel as foreground or background before noising it.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture deliberately showed how simple the corruption step is computationally. It is mostly straightforward numeric manipulation: convert to arrays, add random perturbations, and clip values if needed.",
  },
  {
    id: "mit15773-l11-q35",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements best capture why CLIP was described as conceptually important beyond just this lecture's diffusion use case?",
    options: [
      {
        text: "It showed how text and image embeddings can be learned to inhabit the same conceptual space.",
        isCorrect: true,
      },
      {
        text: "It provided one of the early strong demonstrations of multimodal representation learning.",
        isCorrect: true,
      },
      {
        text: "Its underlying idea is also relevant to modern multimodal large language models.",
        isCorrect: true,
      },
      {
        text: "Its main purpose was presented as replacing cosine similarity with Euclidean distance everywhere in computer vision.",
        isCorrect: false,
      },
    ],
    explanation:
      "CLIP matters not only because it helps diffusion models follow prompts, but because it introduced a powerful general idea: different modalities can be mapped into a shared space where semantic comparison becomes possible. That is foundational for many modern multimodal systems.",
  },
  {
    id: "mit15773-l11-q36",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose a diffusion model is trained only on images of stately college buildings. Which statements are correct?",
    options: [
      {
        text: "Starting from random noise, it should tend to generate different samples that resemble stately college buildings.",
        isCorrect: true,
      },
      {
        text: "The model is intended to learn the target image distribution rather than memorize one fixed output image.",
        isCorrect: true,
      },
      {
        text: "The lecture used this class-specific example to explain how a model can transform noise into samples from a chosen image domain.",
        isCorrect: true,
      },
      {
        text: "The model should frequently generate unrelated categories such as dogs unless the noise itself is manually labeled.",
        isCorrect: false,
      },
    ],
    explanation:
      "The point of the stately-building example was to separate two ideas: randomness creates diversity, while the learned reverse process restricts outputs to the trained domain. The noise does not need labels; the training distribution already teaches the model what kind of images to produce.",
  },
  {
    id: "mit15773-l11-q37",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about steering or conditioning generation with prompts are correct?",
    options: [
      {
        text: "The lecture used 'control,' 'steer,' and 'condition' as near-synonyms for incorporating prompt information into generation.",
        isCorrect: true,
      },
      {
        text: "Conditioning means the model does not simply sample any image from the learned distribution, but one guided by additional information.",
        isCorrect: true,
      },
      {
        text: "A text prompt can therefore bias denoising toward images consistent with that prompt.",
        isCorrect: true,
      },
      {
        text: "Conditioning guarantees that every fine-grained detail of the output must exactly match the user's intended scene with no possible errors.",
        isCorrect: false,
      },
    ],
    explanation:
      "Conditioning narrows and guides the generation process, but it does not make the model perfect. The lecture repeatedly stressed both the power and the limitations of steering: prompts can strongly influence outputs, yet artifacts and inaccuracies can still occur.",
  },
  {
    id: "mit15773-l11-q38",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly interpret the lecture's broader machine-learning lesson from the diffusion setup?",
    options: [
      {
        text: "A clever reformulation of a problem into learnable input-output pairs can make an apparently impossible task tractable.",
        isCorrect: true,
      },
      {
        text: "The diffusion story illustrated a general pattern: once useful supervised pairs are constructed, standard gradient-based training can often be applied.",
        isCorrect: true,
      },
      {
        text: "The success of diffusion models depends partly on turning generation into a sequence of manageable denoising subproblems.",
        isCorrect: true,
      },
      {
        text: "The lecture argued that diffusion is special because it is one of the few machine-learning approaches that does not rely on optimization.",
        isCorrect: false,
      },
    ],
    explanation:
      "One of the clearest conceptual lessons was that a smart problem reformulation can unlock a difficult task. Diffusion does not escape optimization; it makes optimization feasible by turning generation into a learned sequence of easier supervised denoising steps.",
  },
  {
    id: "mit15773-l11-q39",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about the examples shown in the lecture are correct?",
    options: [
      {
        text: "The lecture included examples from ChatGPT, Midjourney, and Sora to motivate the topic.",
        isCorrect: true,
      },
      {
        text: "These examples were used to show that text-to-image and text-to-video systems can produce highly compelling outputs from prompts.",
        isCorrect: true,
      },
      {
        text: "The examples also helped motivate why understanding text-conditional diffusion models matters.",
        isCorrect: true,
      },
      {
        text: "The lecture used these examples mainly to argue that visual generative AI has little relevance outside entertainment.",
        isCorrect: false,
      },
    ],
    explanation:
      "The opening examples were not decorative; they were part of the lecture's argument that diffusion models are a major practical technology. They showed both consumer appeal and the need to understand the modeling ideas behind these systems.",
  },
  {
    id: "mit15773-l11-q40",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly summarize the relationship among diffusion, U-Net, CLIP, attention, and latent space in modern text-to-image systems?",
    options: [
      {
        text: "Diffusion provides the iterative denoising framework for generation.",
        isCorrect: true,
      },
      {
        text: "U-Net provides an image-to-image architecture suitable for denoising transformations.",
        isCorrect: true,
      },
      {
        text: "CLIP provides a way to map prompts into a representation that can guide image generation semantically.",
        isCorrect: true,
      },
      {
        text: "Latent-space processing and attention-based prompt injection help make modern systems both faster and more controllable.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture's final synthesis was that modern systems are combinations of multiple ideas rather than one trick. Diffusion gives the denoising process, U-Net gives the denoising backbone, CLIP aligns language with vision, and latent-space plus attention mechanisms make the whole approach practical and prompt-responsive.",
  },
];
