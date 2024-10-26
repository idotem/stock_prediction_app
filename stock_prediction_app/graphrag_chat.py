import subprocess
import time


def answer_question(question):
    question = '"' + question + '"'
    try:
        result = subprocess.run(['/home/meto/personal-projects/stock_prediction_app/query-graphrag.sh',
                                 question],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True)
        err = result.stderr
        if err:
            print(err)
        # time.sleep(5)
        answer = result.stdout
        answer = answer.rpartition('SUCCESS:')[2]
        answer = answer.split('###')
        return answer
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e.output.decode('utf-8')}"
