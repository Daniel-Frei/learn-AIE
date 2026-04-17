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
        text: "It is not the case that Adding noise requires manually labeling every noisy pixel with a semantic class.",
        isCorrect: true,
      },
    ],
    explanation:
      "In the lecture demo, noise was added by sampling random numbers and adding them to pixel intensities. Because this can push values outside the valid range, clipping is often used, but no semantic labeling of pixels is required for this step.",
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
        text: "It is not the case that The output \\(y\\) must be a class label such as 'building' or 'dog' rather than another image.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture emphasized that the important trick is to make the jump from input to output small. Instead of directly asking the model to go from pure noise to a clean image in one step, we train it on many small denoising transitions between images.",
  },
  {
    id: "mit15773-l11-q04",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Why did the lecture emphasize predicting a 'slightly less noisy' image rather than jumping directly from pure noise to a clean image?",
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
        text: "The lecture suggested that trying to cross the whole gap at once would be too hard for the model.",
        isCorrect: true,
      },
      {
        text: "It is not the case that Using slightly less noisy targets removes the need for any training data.",
        isCorrect: true,
      },
    ],
    explanation:
      "A key intuition in the lecture was that the model should be helped by learning a sequence of easy steps rather than one impossible jump. Training still needs data, but the target is chosen to make the mapping learnable and composable over many iterations.",
  },
  {
    id: "mit15773-l11-q05",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the improved denoising formulation discussed in the lecture?",
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
        text: "This reformulation was presented as producing better image quality in practice.",
        isCorrect: true,
      },
      {
        text: "It is not the case that This reformulation makes the model deterministic only when the training data contains no randomness.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture highlighted a major refinement: predict the noise rather than the partially cleaned image. Subtracting predicted noise from the input is mathematically simple, and in practice this formulation tends to produce better results than directly predicting the less noisy image.",
  },
  {
    id: "mit15773-l11-q06",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the U-Net architecture as presented in the lecture?",
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
        text: "It is not the case that It was presented as the dominant architecture for text tokenization.",
        isCorrect: true,
      },
    ],
    explanation:
      "The U-Net was introduced as a natural architecture for problems where both input and output are images or image-like tensors. The left side compresses and extracts features, while the right side expands back toward the original spatial resolution.",
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
        text: "It is not the case that They are included only so that the number of output channels must equal the number of input channels at every layer.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture emphasized that the cross-connections help the decoder side by reintroducing detailed information from earlier feature maps. This helps preserve spatial detail and reduces the burden on later layers to reconstruct everything from a compressed bottleneck alone.",
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
        text: "It is not the case that The only way to obtain varied outputs is to retrain the model each time with a different dataset.",
        isCorrect: true,
      },
    ],
    explanation:
      "Variation does not require retraining. Once the model has learned how to denoise from random starting points into the target class, different draws from the noise distribution can lead to different valid outputs that all resemble samples from the learned image class.",
  },
  {
    id: "mit15773-l11-q09",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the lecture's intuition for text-controlled image generation?",
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
        text: "The lecture motivated control by saying that a prompt embedding could act like a rough guide toward the kind of image we want.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The lecture claimed that text control works only if we already have the final image and merely need to sharpen it.",
        isCorrect: true,
      },
    ],
    explanation:
      "The prompt is not itself an image, but the lecture argued that it can still guide generation if its embedding is aligned with embeddings of relevant images. In that sense, the text embedding plays the role of a semantic steering signal rather than a literal image template.",
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
        text: "It was introduced in the lecture as a key ingredient for steering diffusion models with text prompts.",
        isCorrect: true,
      },
      {
        text: "It is not the case that Its main purpose in the lecture was to replace noise with a segmentation mask during training.",
        isCorrect: true,
      },
    ],
    explanation:
      "CLIP was presented as a bridge between language and vision. By learning text and image embeddings that reflect the same underlying concept, it becomes possible to use a text prompt to guide a model that ultimately generates images.",
  },
  {
    id: "mit15773-l11-q11",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which requirements on CLIP embeddings were emphasized in the lecture?",
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
        text: "Cosine similarity was used as the notion of closeness.",
        isCorrect: true,
      },
      {
        text: "It is not the case that The desired training objective is achieved by maximizing only the diagonal similarities without penalizing off-diagonal similarities.",
        isCorrect: true,
      },
    ],
    explanation:
      "Maximizing only the matching pairs is not enough because the model could collapse to trivial embeddings. The lecture stressed that CLIP must both pull matching pairs together and push mismatched pairs apart so that the shared space actually carries meaning.",
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
        text: "It is not the case that Because cosine similarity cannot be differentiated, the diagonal-only objective cannot be optimized with gradient descent.",
        isCorrect: true,
      },
    ],
    explanation:
      "The lecture explicitly warned that if we only reward matched pairs, the model can satisfy that objective in a trivial way by making everything similar. Penalizing off-diagonal mismatches prevents this collapse and forces the network to preserve actual semantic distinctions.",
  },
  {
    id: "mit15773-l11-q13",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the CLIP training recipe presented in the lecture?",
    options: [
      {
        text: "It is not the case that A transformer-based text encoder can be used for captions.",

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
        text: "The lecture described CLIP as being trained on only a few thousand hand-labeled image-caption pairs.",
        isCorrect: false,
      },
    ],
    explanation:
      "The CLIP setup uses familiar building blocks: a text encoder and an image encoder. What makes it powerful is the contrastive objective over large batches and an extremely large dataset of image-caption pairs scraped from the internet.",
  },
  {
    id: "mit15773-l11-q14",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about zero-shot image classification with CLIP are correct?",
    options: [
      {
        text: "It is not the case that Candidate class labels can be treated as pieces of text and embedded with CLIP's text encoder.",

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
      "The beauty of CLIP-based zero-shot classification is that labels themselves become text prompts. Because the image and label embeddings live in the same space, one can classify an image by simple similarity comparisons without task-specific retraining.",
  },
  {
    id: "mit15773-l11-q15",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Suppose the denoising model is trained with inputs of the form \\([\\text{image}, \\text{CLIP text embedding of caption}]\\). Which statements are correct?",
    options: [
      {
        text: "It is not the case that The model can learn to use the caption embedding as a conditioning signal during denoising.",

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
      "Adding the text embedding does not replace the diffusion process; it steers it. The model still starts from noise and denoises iteratively, but now it conditions its denoising trajectory on the prompt embedding so that the final image matches the requested concept.",
  },
  {
    id: "mit15773-l11-q16",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe latent diffusion as presented in the lecture?",
    options: [
      {
        text: "It is not the case that Instead of performing diffusion directly in pixel space, the process is carried out in a latent embedding space.",

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
        text: "Latent diffusion was presented as slower than pixel-space diffusion because encoders and decoders add extra layers.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture described latent diffusion as a major speed improvement because operating on compressed latent representations is much cheaper than operating on full-resolution images. The encoder and decoder add steps, but the core iterative denoising becomes far faster.",
  },
  {
    id: "mit15773-l11-q17",
    chapter: 1,
    difficulty: "easy",
    prompt:
      "Which statements about the Hugging Face Diffusers demo in the lecture are correct?",
    options: [
      {
        text: "It is not the case that A pretrained Stable Diffusion pipeline was loaded and used directly from the library.",

        isCorrect: false,
      },
      {
        text: "The prompt 'a photograph of an astronaut riding a horse' was used as a simple text-to-image example.",
        isCorrect: true,
      },
      {
        text: "Changing the random seed led to a different generated image from the same prompt.",
        isCorrect: true,
      },
      {
        text: "The demo showed that diffusion models cannot be accessed through pipeline-style interfaces.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture used the Diffusers library to show how much functionality is already packaged for practitioners. A pretrained pipeline can be loaded and queried with prompts directly, and different random seeds produce different samples for the same textual request.",
  },
  {
    id: "mit15773-l11-q18",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about negative prompting are correct according to the lecture demo?",
    options: [
      {
        text: "It is not the case that It can be used to tell the model what to avoid while still following the main prompt.",

        isCorrect: false,
      },
      {
        text: "The lecture illustrated it by asking for a Labrador in Vermeer style while negatively prompting against blue.",
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
      "Negative prompting is a control trick at inference time, not a retraining method. The lecture used it to show that you can retain a desired style or subject while discouraging some unwanted visual element, such as a color.",
  },
  {
    id: "mit15773-l11-q19",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly reflect the lecture's discussion of limitations and oddities of current text-to-image and text-to-video systems?",
    options: [
      {
        text: "It is not the case that Models may generate anatomically incorrect details such as the wrong number of fingers.",

        isCorrect: false,
      },
      {
        text: "It is not the case that The lecture suggested that these models often do not explicitly encode human concepts like 'five fingers' as symbolic rules.",
        isCorrect: false,
      },
      {
        text: "The apparent realism of generated videos does not necessarily prove that the model has an internal physics engine.",
        isCorrect: true,
      },
      {
        text: "The lecture argued that because these models are trained on many images, they must exactly obey physical laws in all outputs.",
        isCorrect: false,
      },
    ],
    explanation:
      "A major theme was that these systems are astonishingly capable while still fundamentally statistical. They can mimic many regularities seen in the data, but that does not imply explicit symbolic understanding or guaranteed compliance with anatomy or physics.",
  },
  {
    id: "mit15773-l11-q20",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly summarize the broader message of the lecture on diffusion models?",
    options: [
      {
        text: "It is not the case that Diffusion models are not limited to consumer art applications; similar ideas may be useful for scientific design tasks such as proteins.",

        isCorrect: false,
      },
      {
        text: "It is not the case that The main challenge is not only generating an image, but also controlling generation with conditioning information such as text.",
        isCorrect: false,
      },
      {
        text: "Architectures such as U-Net, CLIP, attention mechanisms, and latent-space processing all combine to make modern systems practical.",
        isCorrect: true,
      },
      {
        text: "The lecture concluded that diffusion models are already a solved problem and no longer an active research area.",
        isCorrect: false,
      },
    ],
    explanation:
      "The lecture portrayed diffusion as both practically important and still rapidly evolving. It tied together denoising, conditioning, shared embedding spaces, and latent-space acceleration to explain why modern text-to-image systems are powerful and why the area remains very active.",
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
        text: "It is not the case that The output of one denoising step becomes the input to the next denoising step.",

        isCorrect: false,
      },
      {
        text: "It is not the case that The final image emerges only after a sequence of denoising operations rather than a single direct prediction.",
        isCorrect: false,
      },
      {
        text: "This repeated application is one reason diffusion-based generation can take noticeable time at inference.",
        isCorrect: true,
      },
      {
        text: "The lecture claimed that one denoising step is usually enough once the model is well trained.",
        isCorrect: false,
      },
    ],
    explanation:
      "Diffusion models refine a noisy sample iteratively, and the repeated refinement is an essential part of how they work. The lecture explicitly connected this iterative loop to why text-to-image models often take longer than text models to return a result.",
  },
  {
    id: "mit15773-l11-q23",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly compare training on pixel images versus training in latent space?",
    options: [
      {
        text: "It is not the case that Operating directly in pixel space means the diffusion process repeatedly manipulates full images.",

        isCorrect: false,
      },
      {
        text: "It is not the case that Operating in latent space means the model denoises compressed image representations rather than raw pixels.",
        isCorrect: false,
      },
      {
        text: "Latent-space diffusion was presented as a major speedup over pixel-space diffusion.",
        isCorrect: true,
      },
      {
        text: "Latent diffusion eliminates the need for an image decoder at the end of the process.",
        isCorrect: false,
      },
    ],
    explanation:
      "Pixel-space diffusion works directly on image tensors, which is computationally expensive. Latent diffusion instead denoises a compressed representation and then decodes it back into an image, making the process much faster while preserving strong generation quality.",
  },
  {
    id: "mit15773-l11-q24",
    chapter: 1,
    difficulty: "medium",
    prompt:
      "Which statements about the role of CLIP in text-conditioned image generation are correct?",
    options: [
      {
        text: "It is not the case that CLIP provides text embeddings intended to align with embeddings of semantically matching images.",

        isCorrect: false,
      },
      {
        text: "It is not the case that The conditioning signal for a prompt can therefore guide denoising toward images compatible with that prompt.",
        isCorrect: false,
      },
      {
        text: "The lecture used CLIP as the bridge between textual descriptions and the visual concept space used for generation.",
        isCorrect: true,
      },
      {
        text: "CLIP was introduced as a method for replacing U-Net entirely in diffusion models.",
        isCorrect: false,
      },
    ],
    explanation:
      "CLIP does not replace the denoising network. Instead, it gives a way to represent prompts in a shared concept space with images, which makes text-guided image generation possible.",
  },
  {
    id: "mit15773-l11-q25",
    chapter: 1,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe how CLIP avoids trivial collapse during training?",
    options: [
      {
        text: "It is not the case that If only matching image-caption pairs were rewarded, the model could map everything to nearly the same embedding.",

        isCorrect: false,
      },
      {
        text: "It is not the case that To avoid collapse, the training objective must also penalize similarity for mismatched image-caption pairs.",
        isCorrect: false,
      },
      {
        text: "This forces the model to encode distinctions rather than only universal similarity.",
        isCorrect: true,
      },
      {
        text: "The lecture argued that trivial collapse is impossible whenever cosine similarity is used.",
        isCorrect: false,
      },
    ],
    explanation:
      "Cosine similarity by itself does not prevent collapse. The critical point is the contrastive objective: matched pairs are pulled together and mismatched pairs are pushed apart, which prevents the degenerate solution where every embedding becomes the same.",
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
