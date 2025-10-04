import numpy as np

# Number of students & subjects
students = 30
subjects = 5

# Generate random marks (0–100)
marks = np.random.randint(0, 101, size=(students, subjects))

print("Student Marks:\n", marks)

# Average marks per student
student_avg = np.mean(marks, axis=1)

# Average marks per subject
subject_avg = np.mean(marks, axis=0)

# Overall statistics
overall_mean = np.mean(marks)
overall_median = np.median(marks)
overall_std = np.std(marks)

print("\n📊 Statistics:")
print("Overall Mean:", round(overall_mean, 2))
print("Overall Median:", overall_median)
print("Overall Std Deviation:", round(overall_std, 2))

# Pass/Fail analysis (Pass if avg >= 40)
pass_students = np.sum(student_avg >= 40)
fail_students = students - pass_students
print("\n✅ Passed:", pass_students, " | ❌ Failed:", fail_students)

# Top 5 performers
top5_indices = np.argsort(student_avg)[-5:][::-1]
print("\n🏆 Top 5 Students (by avg marks):")
for i, idx in enumerate(top5_indices, start=1):
    print(f"{i}. Student {idx+1} → Average: {round(student_avg[idx],2)}")

# Subject-wise performance
print("\n📘 Subject-wise Average Marks:", subject_avg)

