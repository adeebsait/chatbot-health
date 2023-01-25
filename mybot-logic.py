#######################################################
#  Initialise NLTK Inference
#######################################################
from nltk.sem import Expression
from nltk.inference import ResolutionProver

read_expr = Expression.fromstring

#######################################################
#  Initialise Knowledgebase.
#######################################################
import pandas

kb = []
data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

# Check if the KB is consistent
inconsistent = ResolutionProver().prove(None, kb)
if inconsistent:
    print("KB is inconsistent!")
    exit(1)

#######################################################
#  Initialise AIML agent
#######################################################
import aiml

# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-logic.xml")

#######################################################
# Welcome user
#######################################################
print("Welcome to this chat bot. Please feel free to ask questions from me!")

#######################################################
# Main loop
#######################################################
while True:
    # get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    # pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    # activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    # post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        # >> YOU already had some other "if" blocks here from the previous
        # courseworks which are not shown here.

        # Here are the processing of the new logical component:
        elif cmd == 31:  # if input pattern is "I know that * is *"
            object, subject = params[1].split(' is ')
            expr = read_expr(subject + '(' + object + ')')
            # check if expr contradicts with the KB before appending
            inconsistent = ResolutionProver().prove(None, kb + [expr])
            if inconsistent:
                print("Sorry, that statement contradicts with the current knowledge base.")
            else:
                kb.append(expr)
                print('OK, I will remember that', object, 'is', subject)
        elif cmd == 32:  # if the input pattern is "check that * is *"
            object, subject = params[1].split(' is ')
            expr = read_expr(subject + '(' + object + ')')
            answer = ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
                print('Correct.')
            else:
                print('It may not be true.')
                # check if the statement is false or not in the knowledge base
                false_expr = read_expr('~' + expr)
                false_answer = ResolutionProver().prove(false_expr, kb, verbose=True)
                if false_answer:
                    print("Incorrect.")
                else:
                    print("Sorry, I don't know.")
        elif cmd == 99:
            print("I did not get that, please try again.")
