import inquirer

def get_user_input():
    # Define the initial question to ask the user
    initial_question = [
        inquirer.List(
            "action",
            message="What would you like to do?",
            choices=["Predict", "Exit"],
        )
    ]

    # Define the questions to ask the user for the prediction
    feature_questions = [
        inquirer.Text("Date", message="Enter Date (YYYY.MM)"),
    ]

    initial_answer = inquirer.prompt(initial_question)

    if initial_answer["action"] == "Predict":
        while True:
            feature_answers = inquirer.prompt(feature_questions)

            # Validate numeric inputs
            try:
                Date = float(feature_answers["Date"])
            except ValueError:
                print(
                    "Please enter valid numeric values for Date"
                )
                continue

            feature_answers["Date"] = Date

            return feature_answers
    else:
        return None