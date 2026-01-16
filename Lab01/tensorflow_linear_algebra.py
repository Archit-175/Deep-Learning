"""
Task 2: TensorFlow Linear Algebra Operations
Demonstration of various linear algebra operations using TensorFlow
"""

import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)
print("\n" + "="*70)
print("TENSORFLOW LINEAR ALGEBRA OPERATIONS")
print("="*70 + "\n")

# 1. Matrix Creation
print("1. MATRIX CREATION")
print("-" * 50)
matrix_a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
matrix_b = tf.constant([[7, 8], [9, 10], [11, 12]], dtype=tf.float32)
vector = tf.constant([1, 2, 3], dtype=tf.float32)

print("Matrix A (2x3):")
print(matrix_a.numpy())
print("\nMatrix B (3x2):")
print(matrix_b.numpy())
print("\nVector (3,):")
print(vector.numpy())

# 2. Matrix Multiplication
print("\n\n2. MATRIX MULTIPLICATION")
print("-" * 50)
result_matmul = tf.matmul(matrix_a, matrix_b)
print("A @ B =")
print(result_matmul.numpy())

# 3. Element-wise Operations
print("\n\n3. ELEMENT-WISE OPERATIONS")
print("-" * 50)
matrix_c = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
matrix_d = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

print("Matrix C:")
print(matrix_c.numpy())
print("\nMatrix D:")
print(matrix_d.numpy())

print("\nC + D (Addition):")
print(tf.add(matrix_c, matrix_d).numpy())

print("\nC * D (Element-wise multiplication):")
print(tf.multiply(matrix_c, matrix_d).numpy())

print("\nC - D (Subtraction):")
print(tf.subtract(matrix_c, matrix_d).numpy())

print("\nC / D (Division):")
print(tf.divide(matrix_c, matrix_d).numpy())

# 4. Matrix Transpose
print("\n\n4. MATRIX TRANSPOSE")
print("-" * 50)
print("Original Matrix A:")
print(matrix_a.numpy())
print("\nTranspose of A:")
print(tf.transpose(matrix_a).numpy())

# 5. Matrix Determinant
print("\n\n5. MATRIX DETERMINANT")
print("-" * 50)
square_matrix = tf.constant([[4, 3], [2, 1]], dtype=tf.float32)
print("Square Matrix:")
print(square_matrix.numpy())
det = tf.linalg.det(square_matrix)
print(f"\nDeterminant: {det.numpy()}")

# 6. Matrix Inverse
print("\n\n6. MATRIX INVERSE")
print("-" * 50)
print("Original Matrix:")
print(square_matrix.numpy())
inverse = tf.linalg.inv(square_matrix)
print("\nInverse Matrix:")
print(inverse.numpy())
print("\nVerification (Matrix * Inverse should give Identity):")
verification = tf.matmul(square_matrix, inverse)
print(verification.numpy())

# 7. Eigenvalues and Eigenvectors
print("\n\n7. EIGENVALUES AND EIGENVECTORS")
print("-" * 50)
symmetric_matrix = tf.constant([[4, 2], [2, 3]], dtype=tf.float32)
print("Symmetric Matrix:")
print(symmetric_matrix.numpy())
eigenvalues, eigenvectors = tf.linalg.eigh(symmetric_matrix)
print("\nEigenvalues:")
print(eigenvalues.numpy())
print("\nEigenvectors:")
print(eigenvectors.numpy())

# 8. Matrix Norm
print("\n\n8. MATRIX NORM")
print("-" * 50)
print("Matrix C:")
print(matrix_c.numpy())
frobenius_norm = tf.norm(matrix_c)
print(f"\nFrobenius Norm: {frobenius_norm.numpy()}")
l1_norm = tf.norm(matrix_c, ord=1)
print(f"L1 Norm: {l1_norm.numpy()}")
l2_norm = tf.norm(matrix_c, ord=2)
print(f"L2 Norm: {l2_norm.numpy()}")

# 9. Matrix Trace
print("\n\n9. MATRIX TRACE")
print("-" * 50)
print("Matrix C:")
print(matrix_c.numpy())
trace = tf.linalg.trace(matrix_c)
print(f"\nTrace (sum of diagonal elements): {trace.numpy()}")

# 10. Matrix Rank
print("\n\n10. MATRIX RANK")
print("-" * 50)
rank_matrix = tf.constant([[1, 2, 3], [2, 4, 6], [3, 6, 9]], dtype=tf.float32)
print("Matrix:")
print(rank_matrix.numpy())
# Note: TensorFlow doesn't have direct rank function, we use SVD
s = tf.linalg.svd(rank_matrix, compute_uv=False)
rank = tf.reduce_sum(tf.cast(s > 1e-10, tf.int32))
print(f"\nMatrix Rank: {rank.numpy()}")

# 11. QR Decomposition
print("\n\n11. QR DECOMPOSITION")
print("-" * 50)
qr_matrix = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
print("Original Matrix:")
print(qr_matrix.numpy())
q, r = tf.linalg.qr(qr_matrix)
print("\nQ (Orthogonal Matrix):")
print(q.numpy())
print("\nR (Upper Triangular Matrix):")
print(r.numpy())
print("\nVerification (Q @ R should give original matrix):")
print(tf.matmul(q, r).numpy())

# 12. Singular Value Decomposition (SVD)
print("\n\n12. SINGULAR VALUE DECOMPOSITION (SVD)")
print("-" * 50)
svd_matrix = tf.constant([[1, 0, 0, 0, 2],
                          [0, 0, 3, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 4, 0, 0, 0]], dtype=tf.float32)
print("Original Matrix:")
print(svd_matrix.numpy())
s, u, v = tf.linalg.svd(svd_matrix)
print("\nSingular Values:")
print(s.numpy())

# 13. Solving Linear Systems (Ax = b)
print("\n\n13. SOLVING LINEAR SYSTEMS (Ax = b)")
print("-" * 50)
A = tf.constant([[3, 1], [1, 2]], dtype=tf.float32)
b = tf.constant([[9], [8]], dtype=tf.float32)
print("Matrix A:")
print(A.numpy())
print("\nVector b:")
print(b.numpy())
x = tf.linalg.solve(A, b)
print("\nSolution x:")
print(x.numpy())
print("\nVerification (A @ x should give b):")
print(tf.matmul(A, x).numpy())

# 14. Cholesky Decomposition
print("\n\n14. CHOLESKY DECOMPOSITION")
print("-" * 50)
pos_def_matrix = tf.constant([[4, 2], [2, 3]], dtype=tf.float32)
print("Positive Definite Matrix:")
print(pos_def_matrix.numpy())
L = tf.linalg.cholesky(pos_def_matrix)
print("\nCholesky Factor L (Lower Triangular):")
print(L.numpy())
print("\nVerification (L @ L^T should give original matrix):")
print(tf.matmul(L, tf.transpose(L)).numpy())

# 15. Matrix Power
print("\n\n15. MATRIX POWER")
print("-" * 50)
power_matrix = tf.constant([[2, 0], [0, 3]], dtype=tf.float32)
print("Original Matrix:")
print(power_matrix.numpy())
power_2 = tf.linalg.matmul(power_matrix, power_matrix)
print("\nMatrix squared:")
print(power_2.numpy())

# 16. Dot Product
print("\n\n16. DOT PRODUCT")
print("-" * 50)
vec1 = tf.constant([1, 2, 3], dtype=tf.float32)
vec2 = tf.constant([4, 5, 6], dtype=tf.float32)
print(f"Vector 1: {vec1.numpy()}")
print(f"Vector 2: {vec2.numpy()}")
dot_product = tf.reduce_sum(tf.multiply(vec1, vec2))
print(f"\nDot Product: {dot_product.numpy()}")

# 17. Cross Product
print("\n\n17. CROSS PRODUCT")
print("-" * 50)
vec3 = tf.constant([1, 0, 0], dtype=tf.float32)
vec4 = tf.constant([0, 1, 0], dtype=tf.float32)
print(f"Vector 1: {vec3.numpy()}")
print(f"Vector 2: {vec4.numpy()}")
cross_product = tf.linalg.cross(vec3, vec4)
print(f"\nCross Product: {cross_product.numpy()}")

# 18. Matrix Concatenation
print("\n\n18. MATRIX CONCATENATION")
print("-" * 50)
mat1 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
mat2 = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
print("Matrix 1:")
print(mat1.numpy())
print("\nMatrix 2:")
print(mat2.numpy())
print("\nVertical Concatenation:")
print(tf.concat([mat1, mat2], axis=0).numpy())
print("\nHorizontal Concatenation:")
print(tf.concat([mat1, mat2], axis=1).numpy())

# 19. Batch Matrix Multiplication
print("\n\n19. BATCH MATRIX MULTIPLICATION")
print("-" * 50)
batch_a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)
batch_b = tf.constant([[[1, 0], [0, 1]], [[2, 0], [0, 2]]], dtype=tf.float32)
print("Batch A shape:", batch_a.shape)
print("Batch B shape:", batch_b.shape)
batch_result = tf.matmul(batch_a, batch_b)
print("\nBatch Matrix Multiplication Result:")
print(batch_result.numpy())

# 20. Matrix Diagonal
print("\n\n20. MATRIX DIAGONAL")
print("-" * 50)
diag_matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
print("Matrix:")
print(diag_matrix.numpy())
diagonal = tf.linalg.diag_part(diag_matrix)
print("\nDiagonal Elements:")
print(diagonal.numpy())

print("\n" + "="*70)
print("All linear algebra operations completed successfully!")
print("="*70)
