import os

current_path = os.path.abspath(os.path.dirname(__file__))

data_path = None # Data path should be inserted here
task_1_data_path = os.path.join(data_path, "Task1", "EASY-RESECT", "NIFTI")
task_2_data_path = os.path.join(data_path, "Task2", "training")
task_3_data_path = os.path.join(data_path, "Task3", "Training")
task_4_data_path = os.path.join(data_path, "Task4", "Training")

test_task_1_data_path = os.path.join(data_path, "Test", "Task1", "NIFTI")
test_task_2_data_path = os.path.join(data_path, "Test", "Task2", "testData")
test_task_3_data_path = os.path.join(data_path, "Test", "Task3", "Testing")
test_task_4_data_path = os.path.join(data_path, "Test", "Task4")

models_path = os.path.join(current_path, "models")