import { Question } from "../../quiz";

export const CrashCourseLinearAlgebraLecture0Questions: Question[] = [
  {
    id: "la-crash-l0-q01",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe the vector \\(v=\\begin{bmatrix}3 \\\\ -4\\end{bmatrix}\\)?",
    options: [
      {
        text: "It can be viewed as a point with coordinates \\((3,-4)\\).",
        isCorrect: true,
      },
      {
        text: "It can be viewed as a displacement \\(3\\) units right and \\(4\\) units down.",
        isCorrect: true,
      },
      {
        text: "Its components are \\(v_1=3\\) and \\(v_2=-4\\).",
        isCorrect: true,
      },
      {
        text: "It has a geometric length \\(\\|v\\|\\) that can be computed from its components.",
        isCorrect: true,
      },
    ],
    explanation:
      "A 2D vector can be interpreted as a coordinate point or as a displacement from the origin. The entries 3 and -4 are its components, and those components determine its direction and magnitude. In this case the negative second component means the vector points downward relative to the positive y-axis.",
  },

  {
    id: "la-crash-l0-q02",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "For vectors \\(a=\\begin{bmatrix}2 \\\\ 5\\end{bmatrix}\\) and \\(b=\\begin{bmatrix}-1 \\\\ 3\\end{bmatrix}\\), which statements are correct?",
    options: [
      {
        text: "\\(a+b=\\begin{bmatrix}1 \\\\ 8\\end{bmatrix}\\).",
        isCorrect: true,
      },
      {
        text: "\\(a-b=\\begin{bmatrix}3 \\\\ 2\\end{bmatrix}\\).",
        isCorrect: true,
      },
      {
        text: "\\(a+b=\\begin{bmatrix}3 \\\\ 2\\end{bmatrix}\\).",
        isCorrect: false,
      },
      {
        text: "Vector addition multiplies corresponding components.",
        isCorrect: false,
      },
    ],
    explanation:
      "Vector addition and subtraction are performed component by component. Adding gives \\([2+(-1),5+3]^T=[1,8]^T\\), and subtracting gives \\([2-(-1),5-3]^T=[3,2]^T\\). Multiplying corresponding components is a different operation and is not vector addition.",
  },

  {
    id: "la-crash-l0-q03",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "What is the Euclidean length of \\(v=\\begin{bmatrix}6 \\\\ 8\\end{bmatrix}\\)?",
    options: [
      { text: "\\(\\sqrt{6^2+8^2}=10\\)", isCorrect: true },
      { text: "\\(6+8=14\\)", isCorrect: false },
      { text: "\\(6\\cdot8=48\\)", isCorrect: false },
      { text: "\\(\\sqrt{14}\\)", isCorrect: false },
    ],
    explanation:
      "The Euclidean length is \\(\\sqrt{6^2+8^2}=\\sqrt{36+64}=\\sqrt{100}=10\\). The value 14 is the sum of the components, not the Euclidean norm. The value 48 is the product of the components, and \\(\\sqrt{14}\\) comes from incorrectly adding the components before taking the square root.",
  },

  {
    id: "la-crash-l0-q04",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe multiplying \\(v=\\begin{bmatrix}2 \\\\ -1\\end{bmatrix}\\) by the scalar \\(-3\\)?",
    options: [
      {
        text: "The result is \\(\\begin{bmatrix}-6 \\\\ 3\\end{bmatrix}\\).",
        isCorrect: true,
      },
      {
        text: "The length is multiplied by \\(|-3|=3\\).",
        isCorrect: true,
      },
      {
        text: "The direction is reversed because the scalar \\(-3\\) is negative.",
        isCorrect: true,
      },
      {
        text: "The result keeps exactly the same direction as \\(v\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Scalar multiplication multiplies every component by the scalar, so \\(-3[2,-1]^T=[-6,3]^T\\). A negative scalar reverses direction, while the absolute value of the scalar multiplies the length. The result is on the same line as the original vector but points the opposite way.",
  },

  {
    id: "la-crash-l0-q05",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "Which statements about the zero vector \\(0=\\begin{bmatrix}0 \\\\ 0\\end{bmatrix}\\) are correct?",
    options: [
      { text: "Its magnitude is 0.", isCorrect: true },
      {
        text: "It has no well-defined direction.",
        isCorrect: true,
      },
      {
        text: "It has magnitude 1 because it is a unit vector.",
        isCorrect: false,
      },
      {
        text: "It points in multiple ordinary vector directions at once.",
        isCorrect: false,
      },
    ],
    explanation:
      "The zero vector has length 0 because all of its components are zero. Since it has no nonzero displacement from the origin, it does not have a well-defined direction. A unit vector must have length 1, so the zero vector is not a unit vector.",
  },

  {
    id: "la-crash-l0-q06",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "For a nonzero vector \\(v\\), which statements correctly describe the unit vector \\(\\hat{v}=\\frac{v}{\\|v\\|}\\)?",
    options: [
      {
        text: "\\(\\hat{v}\\) points in the same direction as \\(v\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\hat{v}\\) has length 1.",
        isCorrect: true,
      },
      {
        text: "Dividing by \\(\\|v\\|\\) removes the magnitude while keeping the direction.",
        isCorrect: true,
      },
      {
        text: "The formula is defined only when \\(v\\) is not the zero vector.",
        isCorrect: true,
      },
    ],
    explanation:
      "A unit vector keeps the direction of the original nonzero vector and rescales its length to 1. Dividing by \\(\\|v\\|\\) removes the magnitude because the vector is scaled by the reciprocal of its own length. The zero vector cannot be normalized because division by zero is undefined.",
  },

  {
    id: "la-crash-l0-q07",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "Which vector is a unit vector pointing in the same direction as \\(v=\\begin{bmatrix}3 \\\\ 4\\end{bmatrix}\\)?",
    options: [
      {
        text: "\\(\\begin{bmatrix}3/5 \\\\ 4/5\\end{bmatrix}\\)",
        isCorrect: true,
      },
      {
        text: "\\(\\begin{bmatrix}5 \\\\ 5\\end{bmatrix}\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\begin{bmatrix}4/5 \\\\ 3/5\\end{bmatrix}\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\begin{bmatrix}-3/5 \\\\ -4/5\\end{bmatrix}\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "The length of \\([3,4]^T\\) is 5, so the unit vector in the same direction is \\([3/5,4/5]^T\\). Swapping the components changes the direction. Negating both components produces a unit vector in the opposite direction.",
  },

  {
    id: "la-crash-l0-q08",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "Which statements about the dot product \\(a\\cdot b\\) are correct?",
    options: [
      {
        text: "It is computed as \\(\\sum_i a_i b_i\\), multiplying corresponding components and adding the results.",
        isCorrect: true,
      },
      {
        text: "It produces a scalar, not a vector.",
        isCorrect: true,
      },
      {
        text: "For real vectors, \\(a\\cdot b=b\\cdot a\\).",
        isCorrect: true,
      },
      {
        text: "It is found by adding the vectors component by component to get \\(a+b\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The dot product multiplies matching components and then sums those products, producing one scalar value. For real vectors, the order of the two vectors does not change the result. Adding vectors component by component produces another vector, so that is not the dot product.",
  },

  {
    id: "la-crash-l0-q09",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "Treat \\(a\\cdot b\\) as the Euclidean dot product \\(a^\\top b\\) for \\(a=\\begin{bmatrix}1 \\\\ 2 \\\\ 3\\end{bmatrix}\\) and \\(b=\\begin{bmatrix}4 \\\\ 0 \\\\ -1\\end{bmatrix}\\). Which statements are correct?",
    options: [
      {
        text: "\\(a^\\top b=1\\cdot4+2\\cdot0+3\\cdot(-1)\\).",
        isCorrect: true,
      },
      { text: "\\(a\\cdot b=1\\).", isCorrect: true },
      { text: "\\(a\\cdot b=7\\).", isCorrect: false },
      {
        text: "The dot product is undefined unless one vector is written horizontally in the prompt.",
        isCorrect: false,
      },
    ],
    explanation:
      "For column vectors, the matrix-product notation for the dot product is \\(a^\\top b\\), so one vector is transposed conceptually even if the shorthand \\(a\\cdot b\\) is used. The calculation is \\(1\\cdot4+2\\cdot0+3\\cdot(-1)=4+0-3=1\\); the value 7 comes from ignoring the negative sign or using the wrong operation.",
  },

  {
    id: "la-crash-l0-q10",
    chapter: 0,
    difficulty: "easy",
    prompt: "Which pair of vectors is orthogonal in \\(\\mathbb{R}^2\\)?",
    options: [
      {
        text: "\\(\\begin{bmatrix}1 \\\\ 2\\end{bmatrix}\\) and \\(\\begin{bmatrix}2 \\\\ -1\\end{bmatrix}\\)",
        isCorrect: true,
      },
      {
        text: "\\(\\begin{bmatrix}1 \\\\ 1\\end{bmatrix}\\) and \\(\\begin{bmatrix}2 \\\\ 2\\end{bmatrix}\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\begin{bmatrix}0 \\\\ 1\\end{bmatrix}\\) and \\(\\begin{bmatrix}0 \\\\ 3\\end{bmatrix}\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\begin{bmatrix}3 \\\\ 4\\end{bmatrix}\\) and \\(\\begin{bmatrix}6 \\\\ 8\\end{bmatrix}\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "Two nonzero vectors are orthogonal when their dot product is zero. The dot product \\([1,2]^T\\cdot[2,-1]^T=2-2=0\\), so that pair is orthogonal. The other pairs point in the same or similar directions and have nonzero dot products.",
  },

  {
    id: "la-crash-l0-q11",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements correctly connect the dot product with angles?",
    options: [
      {
        text: "If \\(a\\cdot b>0\\), then \\(\\cos\\theta>0\\) and the angle between nonzero vectors is acute.",
        isCorrect: true,
      },
      {
        text: "If \\(a\\cdot b=0\\), then \\(\\cos\\theta=0\\) and nonzero vectors are perpendicular.",
        isCorrect: true,
      },
      {
        text: "If \\(a\\cdot b<0\\), then \\(\\cos\\theta<0\\) and the angle between nonzero vectors is obtuse.",
        isCorrect: true,
      },
      {
        text: "\\(a\\cdot b=\\|a\\|\\|b\\|\\cos\\theta\\) for nonzero vectors with angle \\(\\theta\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The formula \\(a\\cdot b=\\|a\\|\\|b\\|\\cos\\theta\\) connects the dot product to the angle between nonzero vectors. Positive, zero, and negative dot products correspond to acute, right, and obtuse angles because the cosine has those signs. This is why the dot product is useful for reasoning about direction and alignment.",
  },

  {
    id: "la-crash-l0-q12",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the scalar projection of \\(a\\) onto a nonzero vector \\(b\\)?",
    options: [
      {
        text: "It measures how much of \\(a\\) points in the direction of \\(b\\).",
        isCorrect: true,
      },
      {
        text: "It can be computed as \\(\\frac{a\\cdot b}{\\|b\\|}\\).",
        isCorrect: true,
      },
      {
        text: "It can be negative when \\(a\\) points partly opposite to \\(b\\).",
        isCorrect: true,
      },
      {
        text: "It equals \\(\\|a\\|\\) even when directions differ.",
        isCorrect: false,
      },
    ],
    explanation:
      "The scalar projection tells how far \\(a\\) extends along the direction of \\(b\\). Its formula is \\((a\\cdot b)/\\|b\\|\\), so the sign depends on whether \\(a\\) is aligned or anti-aligned with \\(b\\). It equals \\(\\|a\\|\\) only in the special case where \\(a\\) points exactly in the same direction as \\(b\\).",
  },

  {
    id: "la-crash-l0-q13",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "For \\(a=\\begin{bmatrix}2 \\\\ 3\\end{bmatrix}\\) and \\(b=\\begin{bmatrix}4 \\\\ 0\\end{bmatrix}\\), which statements are correct?",
    options: [
      {
        text: "The scalar projection of \\(a\\) onto \\(b\\) is \\(\\frac{a\\cdot b}{\\|b\\|}=2\\).",
        isCorrect: true,
      },
      {
        text: "The vector projection of \\(a\\) onto \\(b\\) is \\(\\begin{bmatrix}2 \\\\ 0\\end{bmatrix}\\).",
        isCorrect: true,
      },
      {
        text: "The projection keeps the y-component \\(3\\), giving \\(\\begin{bmatrix}2 \\\\ 3\\end{bmatrix}\\).",
        isCorrect: false,
      },
      {
        text: "The projection is undefined because \\(b\\) has a zero second component.",
        isCorrect: false,
      },
    ],
    explanation:
      "\\(a\\cdot b=8\\) and \\(\\|b\\|=4\\), so the scalar projection is \\(8/4=2\\). The vector projection lies on the x-axis in the direction of \\(b\\), giving \\([2,0]^T\\). A zero component in \\(b\\) is fine as long as \\(b\\) is not the zero vector.",
  },

  {
    id: "la-crash-l0-q14",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statement correctly describes two nonzero vectors in \\(\\mathbb{R}^2\\) that are scalar multiples of each other?",
    options: [
      {
        text: "They lie on the same line through the origin.",
        isCorrect: true,
      },
      {
        text: "They must be perpendicular.",
        isCorrect: false,
      },
      {
        text: "They must have the same length.",
        isCorrect: false,
      },
      {
        text: "They automatically span the entire plane.",
        isCorrect: false,
      },
    ],
    explanation:
      "Nonzero scalar multiples point along the same line through the origin, possibly in opposite directions. They do not have to have the same length because the scalar can stretch or shrink the vector. Since they give only one independent direction, they do not span all of \\(\\mathbb{R}^2\\).",
  },

  {
    id: "la-crash-l0-q15",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements correctly describe linear combinations?",
    options: [
      {
        text: "\\(3u-2v\\) is a linear combination of \\(u\\) and \\(v\\).",
        isCorrect: true,
      },
      {
        text: "A linear combination has the form \\(\\alpha u+\\beta v\\) using scalar multiples and vector addition.",
        isCorrect: true,
      },
      {
        text: "The span of \\(u\\) and \\(v\\) is the set of all vectors \\(\\alpha u+\\beta v\\).",
        isCorrect: true,
      },
      {
        text: "\\(0u+0v\\) is a valid linear combination.",
        isCorrect: true,
      },
    ],
    explanation:
      "A linear combination is formed by multiplying vectors by scalars and adding the results. The expression \\(3u-2v\\) and the zero combination \\(0u+0v\\) are both valid examples. The span is the complete set of vectors reachable through all such combinations.",
  },

  {
    id: "la-crash-l0-q16",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements about the vectors \\(u=\\begin{bmatrix}1 \\\\ 2\\end{bmatrix}\\) and \\(v=\\begin{bmatrix}2 \\\\ 4\\end{bmatrix}\\) are correct?",
    options: [
      {
        text: "\\(v=2u\\), so the vectors are linearly dependent.",
        isCorrect: true,
      },
      {
        text: "Their span is a line through the origin, not the whole plane.",
        isCorrect: true,
      },
      {
        text: "The two vectors point in perpendicular directions.",
        isCorrect: false,
      },
      {
        text: "The two vectors form a basis for \\(\\mathbb{R}^2\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "Because \\(v=2u\\), the two vectors provide only one independent direction. Their span is the line containing \\(u\\), not all of \\(\\mathbb{R}^2\\). They are not perpendicular and cannot form a basis for the plane.",
  },

  {
    id: "la-crash-l0-q17",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements about the standard basis vectors \\(e_1=\\begin{bmatrix}1 \\\\ 0\\end{bmatrix}\\) and \\(e_2=\\begin{bmatrix}0 \\\\ 1\\end{bmatrix}\\) are correct?",
    options: [
      {
        text: "They are linearly independent.",
        isCorrect: true,
      },
      {
        text: "They span \\(\\mathbb{R}^2\\).",
        isCorrect: true,
      },
      {
        text: "Every vector \\(\\begin{bmatrix}x \\\\ y\\end{bmatrix}\\) can be written as \\(xe_1+ye_2\\).",
        isCorrect: true,
      },
      {
        text: "They point in the same direction.",
        isCorrect: false,
      },
    ],
    explanation:
      "The standard basis vectors point along the coordinate axes and are not scalar multiples of each other. Any vector \\([x,y]^T\\) is built from them as \\(xe_1+ye_2\\), so they span the plane. Since they give two independent directions in \\(\\mathbb{R}^2\\), they form a basis.",
  },

  {
    id: "la-crash-l0-q18",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which set is a basis for \\(\\mathbb{R}^2\\)?",
    options: [
      {
        text: "\\(\\left\\{\\begin{bmatrix}1 \\\\ 0\\end{bmatrix},\\begin{bmatrix}1 \\\\ 1\\end{bmatrix}\\right\\}\\)",
        isCorrect: true,
      },
      {
        text: "\\(\\left\\{\\begin{bmatrix}1 \\\\ 2\\end{bmatrix},\\begin{bmatrix}2 \\\\ 4\\end{bmatrix}\\right\\}\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\left\\{\\begin{bmatrix}0 \\\\ 0\\end{bmatrix},\\begin{bmatrix}1 \\\\ 3\\end{bmatrix}\\right\\}\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\left\\{\\begin{bmatrix}5 \\\\ -1\\end{bmatrix}\\right\\}\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "A basis for \\(\\mathbb{R}^2\\) needs two linearly independent vectors. The vectors \\([1,0]^T\\) and \\([1,1]^T\\) are not scalar multiples, so they form a basis. The other sets either contain dependent vectors, include the zero vector, or have too few vectors.",
  },

  {
    id: "la-crash-l0-q19",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe the matrix \\(A=\\begin{bmatrix}1 & 2 & 3 \\\\ 4 & 5 & 6\\end{bmatrix}\\)?",
    options: [
      {
        text: "\\(A\\) has 2 rows and 3 columns.",
        isCorrect: true,
      },
      {
        text: "The entry in row 2, column 3 is 6.",
        isCorrect: true,
      },
      {
        text: "The first row is \\([1\\;2\\;3]\\).",
        isCorrect: true,
      },
      {
        text: "The second column is \\(\\begin{bmatrix}2 \\\\ 5\\end{bmatrix}\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The matrix has two horizontal rows and three vertical columns. Its row 2, column 3 entry is 6, and its first row is \\([1,2,3]\\). The second column is formed from the second entry of each row, giving \\([2,5]^T\\).",
  },

  {
    id: "la-crash-l0-q20",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "For \\(A=\\begin{bmatrix}1 & 2 \\\\ 3 & 4\\end{bmatrix}\\) and \\(x=\\begin{bmatrix}5 \\\\ 6\\end{bmatrix}\\), which statements are correct?",
    options: [
      {
        text: "\\(Ax=\\begin{bmatrix}17 \\\\ 39\\end{bmatrix}\\).",
        isCorrect: true,
      },
      {
        text: "The first output component is the dot product of the first row of \\(A\\) with \\(x\\).",
        isCorrect: true,
      },
      {
        text: "\\(Ax=\\begin{bmatrix}5 \\\\ 6\\end{bmatrix}\\) because multiplying by any matrix leaves a vector unchanged.",
        isCorrect: false,
      },
      {
        text: "The multiplication is invalid because \\(A\\) is square.",
        isCorrect: false,
      },
    ],
    explanation:
      "The first output component is \\(1\\cdot5+2\\cdot6=17\\), and the second is \\(3\\cdot5+4\\cdot6=39\\). Matrix-vector multiplication uses row dot products to create the output vector. A square matrix can multiply a matching vector, and only the identity matrix leaves every vector unchanged.",
  },

  {
    id: "la-crash-l0-q21",
    chapter: 0,
    difficulty: "hard",
    prompt:
      "If \\(A\\) has shape \\(3\\times 2\\) and \\(B\\) has shape \\(2\\times 4\\), what is the shape of \\(AB\\)?",
    options: [
      { text: "\\(3\\times 4\\)", isCorrect: true },
      { text: "\\(2\\times 2\\)", isCorrect: false },
      { text: "\\(4\\times 3\\)", isCorrect: false },
      { text: "The product is undefined.", isCorrect: false },
    ],
    explanation:
      "The inner dimensions 2 and 2 match, so the product \\(AB\\) is defined. The outer dimensions become the shape of the result, giving \\(3\\times4\\). The other shapes come from using the wrong dimensions or incorrectly declaring the product undefined.",
  },

  {
    id: "la-crash-l0-q22",
    chapter: 0,
    difficulty: "hard",
    prompt:
      "Which statements about the identity matrix \\(I_2=\\begin{bmatrix}1 & 0 \\\\ 0 & 1\\end{bmatrix}\\) are correct?",
    options: [
      {
        text: "\\(I_2x=x\\) for every vector \\(x\\in\\mathbb{R}^2\\).",
        isCorrect: true,
      },
      {
        text: "For any \\(2\\times2\\) matrix \\(A\\), multiplying by \\(I_2\\) leaves it unchanged: \\(AI_2=I_2A=A\\).",
        isCorrect: true,
      },
      {
        text: "The identity matrix represents the transformation \\(x\\mapsto x\\) that does nothing.",
        isCorrect: true,
      },
      {
        text: "The identity matrix sends every vector to the zero vector, so \\(I_2x=0\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The identity matrix is the matrix version of doing nothing to a vector or compatible matrix. It preserves each coordinate, so \\(I_2x=x\\). Sending every vector to zero is the zero transformation, not the identity transformation.",
  },

  {
    id: "la-crash-l0-q23",
    chapter: 0,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe the transpose of \\(A=\\begin{bmatrix}1 & 2 & 3 \\\\ 4 & 5 & 6\\end{bmatrix}\\)?",
    options: [
      {
        text: "\\(A^T=\\begin{bmatrix}1 & 4 \\\\ 2 & 5 \\\\ 3 & 6\\end{bmatrix}\\).",
        isCorrect: true,
      },
      {
        text: "\\(A^T\\) has shape \\(3\\times2\\).",
        isCorrect: true,
      },
      {
        text: "Rows of \\(A\\) become columns of \\(A^T\\).",
        isCorrect: true,
      },
      {
        text: "Columns of \\(A\\) become rows of \\(A^T\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "The transpose flips rows and columns, so the \\(2\\times3\\) matrix becomes a \\(3\\times2\\) matrix. The first row \\([1,2,3]\\) becomes the first column of \\(A^T\\), and the columns of \\(A\\) become rows of \\(A^T\\). This operation changes orientation but not the individual entry values.",
  },

  {
    id: "la-crash-l0-q24",
    chapter: 0,
    difficulty: "hard",
    prompt:
      "Which statements about \\(A=\\begin{bmatrix}2 & 1 \\\\ 1 & 2\\end{bmatrix}\\) are correct?",
    options: [
      { text: "\\(A\\) is symmetric.", isCorrect: true },
      {
        text: "\\(A^T=A\\).",
        isCorrect: true,
      },
      {
        text: "\\(A\\) is not symmetric because it has nonzero off-diagonal entries.",
        isCorrect: false,
      },
      {
        text: "\\(A\\) is not square.",
        isCorrect: false,
      },
    ],
    explanation:
      "A matrix is symmetric when it equals its transpose. Here the off-diagonal entries mirror each other because both are 1, so \\(A^T=A\\). A symmetric matrix may have nonzero off-diagonal entries as long as the mirrored entries match.",
  },

  {
    id: "la-crash-l0-q25",
    chapter: 0,
    difficulty: "hard",
    prompt:
      "What is the determinant of \\(A=\\begin{bmatrix}3 & 2 \\\\ 1 & 4\\end{bmatrix}\\)?",
    options: [
      { text: "10", isCorrect: true },
      { text: "14", isCorrect: false },
      { text: "5", isCorrect: false },
      { text: "-10", isCorrect: false },
    ],
    explanation:
      "For a \\(2\\times2\\) matrix \\(\\begin{bmatrix}a & b \\\\ c & d\\end{bmatrix}\\), the determinant is \\(ad-bc\\). Here that gives \\(3\\cdot4-2\\cdot1=12-2=10\\). The other values come from adding products, omitting a term, or using the wrong sign.",
  },

  {
    id: "la-crash-l0-q26",
    chapter: 0,
    difficulty: "hard",
    prompt:
      "Which statements correctly connect a \\(2\\times2\\) determinant to geometry and invertibility?",
    options: [
      {
        text: "A nonzero determinant means the two columns are linearly independent.",
        isCorrect: true,
      },
      {
        text: "The absolute value of the determinant gives the area scale factor for the unit square.",
        isCorrect: true,
      },
      {
        text: "A zero determinant means the transformation collapses area to zero.",
        isCorrect: true,
      },
      {
        text: "A zero determinant means the matrix is invertible.",
        isCorrect: false,
      },
    ],
    explanation:
      "For a \\(2\\times2\\) matrix, a nonzero determinant means the columns provide two independent directions and the transformation is invertible. The determinant's absolute value tells how areas scale under the transformation. If the determinant is zero, area collapses to zero and the matrix cannot be inverted.",
  },

  {
    id: "la-crash-l0-q27",
    chapter: 0,
    difficulty: "hard",
    prompt:
      "For the matrix \\(A=\\begin{bmatrix}1 & 2 \\\\ 2 & 4\\end{bmatrix}\\), which statements are correct?",
    options: [
      {
        text: "The second row is twice the first row.",
        isCorrect: true,
      },
      {
        text: "The rank of \\(A\\) is 1.",
        isCorrect: true,
      },
      {
        text: "The rows give two independent directions.",
        isCorrect: false,
      },
      {
        text: "The determinant of \\(A\\) is nonzero.",
        isCorrect: false,
      },
    ],
    explanation:
      "The second row \\([2,4]\\) is exactly twice the first row \\([1,2]\\), so the rows contain only one independent direction. Therefore the rank is 1 rather than 2. The determinant is \\(1\\cdot4-2\\cdot2=0\\), which agrees with the rank deficiency.",
  },

  {
    id: "la-crash-l0-q28",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements correctly describe the rank of a set of vectors?",
    options: [
      {
        text: "It is the number of independent directions in \\(\\operatorname{span}\\{v_1,\\dots,v_k\\}\\).",
        isCorrect: true,
      },
      {
        text: "Two nonzero scalar multiples such as \\(u\\) and \\(2u\\) have rank 1 as a set.",
        isCorrect: true,
      },
      {
        text: "The set \\(\\{e_1,e_2\\}\\) has rank 2 in \\(\\mathbb{R}^2\\).",
        isCorrect: true,
      },
      {
        text: "Adding the zero vector \\(0\\) to a set increases its rank by 1.",
        isCorrect: false,
      },
    ],
    explanation:
      "Rank counts independent directions, not just how many vectors are listed. Two nonzero scalar multiples provide only one direction, while \\(e_1\\) and \\(e_2\\) provide two independent directions in the plane. The zero vector adds no new direction, so it does not increase rank.",
  },

  {
    id: "la-crash-l0-q29",
    chapter: 0,
    difficulty: "hard",
    prompt:
      "Which row-reduced form indicates a \\(2\\times2\\) system has a unique solution?",
    options: [
      {
        text: "\\(\\begin{bmatrix}1 & 0 \\\\ 0 & 1\\end{bmatrix}\\)",
        isCorrect: true,
      },
      {
        text: "\\(\\begin{bmatrix}1 & 2 \\\\ 0 & 0\\end{bmatrix}\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\begin{bmatrix}0 & 0 \\\\ 0 & 0\\end{bmatrix}\\)",
        isCorrect: false,
      },
      {
        text: "\\(\\begin{bmatrix}1 & 0 \\\\ 0 & 0\\end{bmatrix}\\)",
        isCorrect: false,
      },
    ],
    explanation:
      "A \\(2\\times2\\) coefficient matrix with pivots in both columns has full rank and gives a unique solution for each compatible right-hand side. The identity matrix shows two pivots, one for each variable. The other forms have fewer than two pivots, so they do not represent a fully determined \\(2\\times2\\) system.",
  },

  {
    id: "la-crash-l0-q30",
    chapter: 0,
    difficulty: "hard",
    prompt:
      "Which statements correctly describe solving the system \\(x+2y=5\\) and \\(3x+6y=15\\)?",
    options: [
      {
        text: "The second equation is three times the first equation.",
        isCorrect: true,
      },
      {
        text: "The system has infinitely many solutions along one line.",
        isCorrect: true,
      },
      {
        text: "The coefficient rows are linearly dependent.",
        isCorrect: true,
      },
      {
        text: "The system does not have a unique solution because the equations describe the same line.",
        isCorrect: true,
      },
    ],
    explanation:
      "The second equation is exactly three times the first, so both equations describe the same line. That means there are infinitely many solutions on the line \\(x+2y=5\\), not one unique intersection point. The coefficient rows are linearly dependent because one is a scalar multiple of the other.",
  },

  {
    id: "la-crash-l0-q31",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "Which statements correctly interpret the notation \\(x \\in \\mathbb{R}^n\\)?",
    options: [
      {
        text: "\\(x\\) is a vector with \\(n\\) real-number components.",
        isCorrect: true,
      },
      {
        text: "\\(\\mathbb{R}\\) denotes the set of real numbers.",
        isCorrect: true,
      },
      {
        text: "\\(\\mathbb{R}^n\\) denotes n-dimensional real coordinate space.",
        isCorrect: true,
      },
      {
        text: 'The symbol \\(\\in\\) means "is an element of" or "belongs to".',
        isCorrect: true,
      },
    ],
    explanation:
      "The notation \\(x \\in \\mathbb{R}^n\\) says that \\(x\\) belongs to n-dimensional real coordinate space. That means \\(x\\) has \\(n\\) entries and each entry is a real number. The symbol \\(\\in\\) is membership notation, so it tells what set or space an object belongs to.",
  },

  {
    id: "la-crash-l0-q32",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "For \\(x=\\begin{bmatrix}x_1 \\\\ x_2 \\\\ x_3\\end{bmatrix}\\), which statements are correct?",
    options: [
      {
        text: "\\(x_1\\), \\(x_2\\), and \\(x_3\\) are components of \\(x\\).",
        isCorrect: true,
      },
      {
        text: "The subscripts in \\(x_i\\) label positions within the vector.",
        isCorrect: true,
      },
      {
        text: "The vector has dimension \\(3\\), so \\(x\\in\\mathbb{R}^3\\) if its entries are real.",
        isCorrect: true,
      },
      {
        text: "\\(x_3\\) means \\(x\\) raised to the third power.",
        isCorrect: false,
      },
    ],
    explanation:
      "Subscripts such as \\(x_1\\) and \\(x_3\\) label components or positions in a vector. In this notation, \\(x_3\\) means the third component of \\(x\\), not a power. The displayed column vector has three components because it has three entries.",
  },

  {
    id: "la-crash-l0-q33",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe row and column vector notation?",
    options: [
      {
        text: "\\(\\begin{bmatrix}1 \\\\ 2\\end{bmatrix}\\) is written as a column vector.",
        isCorrect: true,
      },
      {
        text: "\\([1\\;2]\\) is written as a row vector.",
        isCorrect: true,
      },
      {
        text: "A row vector and a column vector have the same orientation.",
        isCorrect: false,
      },
      {
        text: "The notation \\(^T\\) means to add the vector components.",
        isCorrect: false,
      },
    ],
    explanation:
      "A column vector lists entries vertically, while a row vector lists entries horizontally. The superscript \\(^T\\) means transpose, which changes rows into columns and columns into rows. It does not mean summing the vector components.",
  },

  {
    id: "la-crash-l0-q34",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "For a matrix \\(A\\), what does the notation \\(A_{ij}\\) usually mean?",
    options: [
      {
        text: "The entry of \\(A\\) in row \\(i\\), column \\(j\\).",
        isCorrect: true,
      },
      {
        text: "The product of row \\(i\\) and row \\(j\\).",
        isCorrect: false,
      },
      {
        text: "The determinant of \\(A\\).",
        isCorrect: false,
      },
      {
        text: "The transpose of \\(A\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The notation \\(A_{ij}\\) refers to a particular matrix entry: row \\(i\\), column \\(j\\). This is different from determinant notation, transpose notation, or multiplying rows. Subscripts on a matrix are position labels.",
  },

  {
    id: "la-crash-l0-q35",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "Which statements correctly interpret \\(A \\in \\mathbb{R}^{m \\times n}\\)?",
    options: [
      {
        text: "\\(A\\) is a matrix with \\(m\\) rows.",
        isCorrect: true,
      },
      {
        text: "\\(A\\) is a matrix with \\(n\\) columns.",
        isCorrect: true,
      },
      {
        text: "The entries of \\(A\\) are real numbers.",
        isCorrect: true,
      },
      {
        text: "The notation \\(m\\times n\\) describes the matrix shape as rows by columns.",
        isCorrect: true,
      },
    ],
    explanation:
      "The notation \\(A \\in \\mathbb{R}^{m \\times n}\\) says that \\(A\\) is an \\(m\\)-by-\\(n\\) real matrix. The first dimension is the number of rows, and the second dimension is the number of columns. This shape notation is essential for deciding whether matrix products are valid.",
  },

  {
    id: "la-crash-l0-q36",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "Which statements correctly describe common symbols for zero and identity objects?",
    options: [
      {
        text: "\\(0\\) can denote a zero scalar, zero vector, or zero matrix depending on context.",
        isCorrect: true,
      },
      {
        text: "\\(I_n\\) denotes the \\(n\\times n\\) identity matrix.",
        isCorrect: true,
      },
      {
        text: "The identity matrix leaves compatible vectors unchanged, so \\(I_nx=x\\).",
        isCorrect: true,
      },
      {
        text: "A zero matrix and an identity matrix are the same object, so \\(0_{n\\times n}=I_n\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The symbol \\(0\\) is context-dependent, so it can refer to a scalar, vector, or matrix filled with zeros. The identity matrix \\(I_n\\) is an \\(n\\times n\\) matrix that leaves compatible vectors unchanged. A zero matrix sends vectors to zero, so it is not the same as an identity matrix.",
  },

  {
    id: "la-crash-l0-q37",
    chapter: 0,
    difficulty: "easy",
    prompt:
      "Which statements correctly interpret norm notation such as \\(\\|x\\|\\), \\(\\|x\\|_2\\), and \\(\\|x\\|_1\\)?",
    options: [
      {
        text: "\\(\\|x\\|\\) commonly denotes the length or magnitude of a vector.",
        isCorrect: true,
      },
      {
        text: "\\(\\|x\\|_2\\) denotes the Euclidean norm unless another convention is stated.",
        isCorrect: true,
      },
      {
        text: "\\(\\|x\\|_1\\) means the first component of \\(x\\).",
        isCorrect: false,
      },
      {
        text: "Double bars around a vector mean the same thing as matrix transpose.",
        isCorrect: false,
      },
    ],
    explanation:
      "Double bars usually denote a norm, which is a way of measuring vector size. The subscript distinguishes different norms, such as the Euclidean \\(L2\\) norm and the \\(L1\\) norm. A norm subscript is not a component index, and it is unrelated to transpose notation.",
  },

  {
    id: "la-crash-l0-q38",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "What does the summation notation \\(\\sum_{i=1}^{n} x_i y_i\\) mean?",
    options: [
      {
        text: "Add the products \\(x_i y_i\\), giving \\(x_1y_1+x_2y_2+\\cdots+x_ny_n\\).",
        isCorrect: true,
      },
      {
        text: "Multiply the components separately, giving \\((\\prod_i x_i)(\\prod_i y_i)\\).",
        isCorrect: false,
      },
      {
        text: "Choose the largest value, giving \\(\\max_i x_i y_i\\).",
        isCorrect: false,
      },
      {
        text: "Add the components separately, giving \\(\\sum_i x_i+\\sum_i y_i\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "The symbol \\(\\sum\\) means to add a sequence of terms. The expression \\(\\sum_{i=1}^{n} x_i y_i\\) means compute \\(x_1y_1+x_2y_2+\\cdots+x_ny_n\\). It is not a maximum, a product of all components at once, or two separate sums of the components.",
  },

  {
    id: "la-crash-l0-q39",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements correctly describe equivalent dot-product notation for column vectors \\(a,b\\in\\mathbb{R}^n\\)?",
    options: [
      {
        text: "\\(a\\cdot b\\) denotes the dot product.",
        isCorrect: true,
      },
      {
        text: "\\(a^T b\\) is a matrix notation for the same scalar dot product.",
        isCorrect: true,
      },
      {
        text: "\\(\\sum_i a_i b_i\\) is component-sum notation for the same dot product.",
        isCorrect: true,
      },
      {
        text: "All three notations describe multiplying matching components and summing the results.",
        isCorrect: true,
      },
    ],
    explanation:
      "For real column vectors, \\(a\\cdot b\\), \\(a^T b\\), and \\(\\sum_i a_i b_i\\) commonly describe the same dot product. The transpose in \\(a^T b\\) turns \\(a\\) into a row vector so the matrix multiplication produces one scalar. All of these notations encode matching-component multiplication followed by addition.",
  },

  {
    id: "la-crash-l0-q40",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "Which statements correctly interpret the expression \\(x^T y\\) when \\(x\\) and \\(y\\) are column vectors in \\(\\mathbb{R}^n\\)?",
    options: [
      {
        text: "\\(x^T\\) is a row vector.",
        isCorrect: true,
      },
      {
        text: "\\(x^T y\\) is a scalar.",
        isCorrect: true,
      },
      {
        text: "\\(x^T y\\) equals the dot product of \\(x\\) and \\(y\\).",
        isCorrect: true,
      },
      {
        text: "\\(x^T y\\) is a vector with \\(n\\) components.",
        isCorrect: false,
      },
    ],
    explanation:
      "If \\(x\\) and \\(y\\) are column vectors, then \\(x^T\\) is a row vector. Multiplying a \\(1\\times n\\) row vector by an \\(n\\times1\\) column vector gives a \\(1\\times1\\) scalar, which is the dot product. It does not produce a new \\(n\\)-component vector.",
  },

  {
    id: "la-crash-l0-q41",
    chapter: 0,
    difficulty: "medium",
    prompt:
      "If \\(A\\in\\mathbb{R}^{m\\times n}\\) and \\(x\\in\\mathbb{R}^n\\), which statements correctly describe \\(Ax\\)?",
    options: [
      {
        text: "The product is defined because the \\(n\\) columns of \\(A\\) match the \\(n\\) entries of \\(x\\).",
        isCorrect: true,
      },
      {
        text: "\\(Ax\\in\\mathbb{R}^m\\).",
        isCorrect: true,
      },
      {
        text: "\\(Ax\\in\\mathbb{R}^n\\) because \\(x\\) has \\(n\\) components.",
        isCorrect: false,
      },
      {
        text: "The product is defined when \\(m=n\\) rather than when inner dimensions match.",
        isCorrect: false,
      },
    ],
    explanation:
      "The product \\(Ax\\) is defined when the number of matrix columns matches the number of vector entries. The output has one component per row of \\(A\\), so it lies in \\(\\mathbb{R}^m\\). The matrix does not need to be square for matrix-vector multiplication to work.",
  },

  {
    id: "la-crash-l0-q42",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statement correctly interprets \\(ABx=A(Bx)\\)?",
    options: [
      {
        text: "Apply \\(B\\) to \\(x\\) first, then apply \\(A\\) to the result.",
        isCorrect: true,
      },
      {
        text: "Apply \\(A\\) to \\(x\\) first, then apply \\(B\\) to the result.",
        isCorrect: false,
      },
      {
        text: "Multiply corresponding entries of \\(A\\), \\(B\\), and \\(x\\).",
        isCorrect: false,
      },
      {
        text: "The notation means \\(AB=BA\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "In the expression \\(ABx\\), the operation closest to \\(x\\) happens first, so \\(B\\) transforms \\(x\\) before \\(A\\) is applied. This is composition notation. It does not imply element-wise multiplication or that matrix multiplication is commutative.",
  },

  {
    id: "la-crash-l0-q43",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements correctly describe the term span?",
    options: [
      {
        text: "The span of vectors \\(v_1,\\dots,v_k\\) is the set of all linear combinations \\(\\sum_i \\alpha_i v_i\\).",
        isCorrect: true,
      },
      {
        text: "If two non-parallel vectors in \\(\\mathbb{R}^2\\) are given, their span is all of \\(\\mathbb{R}^2\\).",
        isCorrect: true,
      },
      {
        text: "If a single nonzero vector in \\(\\mathbb{R}^2\\) is given, its span is a line through the origin.",
        isCorrect: true,
      },
      {
        text: "The zero vector is included in any span because choosing all coefficients \\(\\alpha_i=0\\) gives \\(0\\).",
        isCorrect: true,
      },
    ],
    explanation:
      "Span means all vectors that can be produced by scalar-multiplying the given vectors and adding the results. One nonzero vector spans a line through the origin, while two independent vectors in the plane span all of \\(\\mathbb{R}^2\\). The zero vector is included because choosing all scalar coefficients to be zero gives the zero vector.",
  },

  {
    id: "la-crash-l0-q44",
    chapter: 0,
    difficulty: "medium",
    prompt: "Which statements correctly describe the term basis?",
    options: [
      {
        text: "A basis for a space is a set of vectors that spans the space and is linearly independent.",
        isCorrect: true,
      },
      {
        text: "A basis gives a coordinate system for writing vectors in that space.",
        isCorrect: true,
      },
      {
        text: "The standard basis for \\(\\mathbb{R}^2\\) is \\(\\{e_1,e_2\\}\\).",
        isCorrect: true,
      },
      {
        text: "A basis for \\(\\mathbb{R}^2\\) can include the zero vector.",
        isCorrect: false,
      },
    ],
    explanation:
      "A basis must both span the space and avoid redundant directions. This lets every vector in the space be written using coordinates relative to that basis. The zero vector cannot be part of a basis because it contributes no independent direction.",
  },

  {
    id: "la-crash-l0-q45",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements correctly interpret rank terminology?",
    options: [
      {
        text: "\\(\\operatorname{rank}(A)\\) counts independent directions represented by rows or columns.",
        isCorrect: true,
      },
      {
        text: "A \\(3\\times5\\) matrix can have rank at most 3.",
        isCorrect: true,
      },
      {
        text: "\\(\\operatorname{rank}(A)\\) is the same as the sum of matrix entries \\(\\sum_{ij}A_{ij}\\).",
        isCorrect: false,
      },
      {
        text: "A matrix with \\(\\operatorname{rank}(A)=0\\) must have at least one nonzero row.",
        isCorrect: false,
      },
    ],
    explanation:
      "Rank measures the number of linearly independent directions in the rows or columns of a matrix. A matrix cannot have rank larger than its smaller dimension, so a \\(3\\times5\\) matrix has rank at most 3. Rank is not an entry sum, and rank 0 means the matrix has no nonzero independent row or column direction.",
  },

  {
    id: "la-crash-l0-q46",
    chapter: 0,
    difficulty: "hard",
    prompt:
      "Which statement correctly connects determinant and inverse notation for a square matrix \\(A\\)?",
    options: [
      {
        text: "If \\(\\det(A)\\ne0\\), then \\(A^{-1}\\) exists.",
        isCorrect: true,
      },
      {
        text: "\\(A^{-1}\\) means each entry of \\(A\\) is negated.",
        isCorrect: false,
      },
      {
        text: "\\(\\det(A)\\) denotes the transpose of \\(A\\).",
        isCorrect: false,
      },
      {
        text: "A determinant is defined for rectangular matrices.",
        isCorrect: false,
      },
    ],
    explanation:
      "For a square matrix, a nonzero determinant means the matrix is invertible and \\(A^{-1}\\) exists. The notation \\(A^{-1}\\) means inverse matrix, not negating entries. The determinant is not the transpose, and standard determinant notation applies to square matrices.",
  },

  {
    id: "la-crash-l0-q47",
    chapter: 0,
    difficulty: "hard",
    prompt: "Which statements correctly describe symmetric matrix notation?",
    options: [
      {
        text: "A symmetric matrix satisfies \\(A=A^T\\).",
        isCorrect: true,
      },
      {
        text: "A symmetric matrix must be square.",
        isCorrect: true,
      },
      {
        text: "For a symmetric matrix, mirrored entries across the main diagonal are equal.",
        isCorrect: true,
      },
      {
        text: "A covariance-style matrix such as \\(X^T X\\) is symmetric.",
        isCorrect: true,
      },
    ],
    explanation:
      "The notation \\(A=A^T\\) defines symmetry: the matrix equals its transpose. This requires a square shape, and entries mirror across the main diagonal. Matrices of the form \\(X^T X\\) are symmetric because transposing the product gives the same matrix.",
  },

  {
    id: "la-crash-l0-q48",
    chapter: 0,
    difficulty: "hard",
    prompt:
      "Which statements correctly interpret eigenvector notation \\(Av=\\lambda v\\)?",
    options: [
      {
        text: "\\(v\\) is an eigenvector when it is nonzero and its direction is preserved by \\(A\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\lambda\\) is the eigenvalue, a scalar scale factor.",
        isCorrect: true,
      },
      {
        text: "The expression says applying \\(A\\) to \\(v\\) gives a scalar multiple of \\(v\\).",
        isCorrect: true,
      },
      {
        text: "The notation means vectors are changed into perpendicular vectors.",
        isCorrect: false,
      },
    ],
    explanation:
      "The equation \\(Av=\\lambda v\\) says that applying \\(A\\) to \\(v\\) changes its length or sign but keeps it on the same line. The scalar \\(\\lambda\\) is the eigenvalue. The vector \\(v\\) must be nonzero, and the notation does not describe a forced perpendicular direction.",
  },

  {
    id: "la-crash-l0-q49",
    chapter: 0,
    difficulty: "hard",
    prompt:
      "Which statements correctly interpret the singular value decomposition notation \\(A=U\\Sigma V^T\\)?",
    options: [
      {
        text: "\\(V^T\\) means the transpose of \\(V\\).",
        isCorrect: true,
      },
      {
        text: "\\(\\Sigma\\) is commonly used for the diagonal matrix of singular values.",
        isCorrect: true,
      },
      {
        text: "The notation means \\(A\\) is the sum \\(U+\\Sigma+V^T\\).",
        isCorrect: false,
      },
      {
        text: "The notation means \\(\\Sigma\\) is a vector with components named \\(\\Sigma_i\\).",
        isCorrect: false,
      },
    ],
    explanation:
      "In singular value decomposition notation, \\(A=U\\Sigma V^T\\) is a matrix product, not a sum. The superscript \\(^T\\) marks the transpose of \\(V\\), and \\(\\Sigma\\) is commonly the diagonal matrix containing singular values. This notation is used to describe a transformation in terms of rotations/reflections and scaling.",
  },

  {
    id: "la-crash-l0-q50",
    chapter: 0,
    difficulty: "hard",
    prompt:
      "Which statement correctly identifies the object types in \\(c\\in\\mathbb{R}\\), \\(v\\in\\mathbb{R}^n\\), and \\(A\\in\\mathbb{R}^{m\\times n}\\)?",
    options: [
      {
        text: "\\(c\\) is a scalar, \\(v\\) is a vector, and \\(A\\) is a matrix.",
        isCorrect: true,
      },
      {
        text: "\\(c\\), \\(v\\), and \\(A\\) are required to be square matrices.",
        isCorrect: false,
      },
      {
        text: "\\(v\\) is a scalar because it has a single letter name.",
        isCorrect: false,
      },
      {
        text: "\\(A\\) is a vector because \\(m\\times n\\) uses two symbols.",
        isCorrect: false,
      },
    ],
    explanation:
      "The notation \\(c\\in\\mathbb{R}\\) identifies \\(c\\) as a real scalar, while \\(v\\in\\mathbb{R}^n\\) identifies \\(v\\) as an n-dimensional vector. The notation \\(A\\in\\mathbb{R}^{m\\times n}\\) identifies \\(A\\) as a matrix with \\(m\\) rows and \\(n\\) columns. Object type comes from the mathematical space, not merely from the letter name.",
  },
];

export const CrashCourseLinearAlgebraL0Questions =
  CrashCourseLinearAlgebraLecture0Questions;
