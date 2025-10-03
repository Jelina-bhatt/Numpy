import numpy as np

# ---------------------------
# 1. Generate Random Data
# ---------------------------
# 10 students, 5 subjects
marks = np.random.randint(40, 101, size=(10, 5))
print("Marks of 10 students in 5 subjects:\n", marks)

# ---------------------------
# 2. Subject-wise Statistics
# ---------------------------
print("\nAverage marks per subject:", np.mean(marks, axis=0))
print("Highest marks per subject:", np.max(marks, axis=0))
print("Lowest marks per subject:", np.min(marks, axis=0))

# ---------------------------
# 3. Student-wise Statistics
# ---------------------------
print("\nTotal marks per student:", np.sum(marks, axis=1))
print("Average marks per student:", np.mean(marks, axis=1))

# ---------------------------
# 4. Broadcasting Example
# ---------------------------
# Suppose we add 5 bonus marks to every subject
bonus = np.array([5, 5, 5, 5, 5])
new_marks = marks + bonus
print("\nMarks after adding bonus:\n", new_marks)

# ---------------------------
# 5. Ranking Students
# ---------------------------
total_marks = np.sum(marks, axis=1)
ranking = np.argsort(-total_marks)  # sort descending
print("\nRanking of students (by index):", ranking)
print("Topper is Student:", ranking[0], "with", total_marks[ranking[0]], "marks")
