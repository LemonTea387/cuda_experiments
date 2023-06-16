import re
import matplotlib.pyplot as plt

RESULT = '^(\d+), (\d+.\d+)'
START_MODE = '^Start Test on (\w+)'
OPERATION = '^Performing (\w+)'

def read_result():
    '''
        Reads result.txt and returns the extracted data points to be used 
        for plotting in a 3 element tuple of lists of dictionary.
        Dictionary will hold the size value as key and time value as value. 
    '''
    # Results list storing GPU, CPU Parallel, CPU Serial times in order
    results_addition = [dict(), dict(), dict()]
    results_multiplication = [dict(), dict(), dict()]
    results_inversion = [dict(), dict(), dict()]
    with open('result.txt', 'r') as f:
        # Cheap way to bind the dictionary and insert the values into it.
        operation = ''
        processor = ''
        operation_store = None
        result_store = None
        for line in f:
            # Check if it's indicating the operation follows
            if m:=re.match(OPERATION, line):
                operation = m.group(1)
                if operation == 'Addition':
                    operation_store = results_addition
                elif operation == 'Multiplication':
                    operation_store = results_multiplication
                elif operation == 'Inversion':
                    operation_store = results_inversion
            
            # Check if it's indicating the start of type of processor
            elif m:=re.match(START_MODE, line):
                processor = m.group(1)
                if processor == 'GPU':
                    result_store = operation_store[0]
                elif processor == 'Parallel':
                    result_store = operation_store[1]
                elif processor == 'Serial':
                    result_store = operation_store[2]
            # Raw data line
            elif m:=re.match(RESULT, line):
                result_store[int(m.group(1))] = float(m.group(2))

    return (results_addition, results_multiplication, results_inversion)

def plot_result(title, data, *labels):
    '''
    Plot the results using matplotlib.
    Takes in a list of data points(dict of xval to yval) with len(data) amount of labels.
    Plot them all in a line graph.
    '''
    if len(labels) != len(data):
        print('Error! Please provide the labels to each data line')
        return -1
    
    plt.xlabel('Square Matrix Single Side Width')
    plt.ylabel('Time(s)')
    plt.title(title)
    for label, x in zip(labels, data):
        # Extract from dict
        xvals = x.keys()
        yval = x.values()
        plt.plot(xvals, yval,label=label)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data = read_result()
    plot_result('Matrix Addition', data[0], 'GPU','CPU Parallel', 'CPU Serial')
    plot_result('Matrix Multiplication', data[1], 'GPU','CPU Parallel', 'CPU Serial')
    plot_result('Matrix Inversion', data[2], 'GPU','CPU Parallel', 'CPU Serial')

    