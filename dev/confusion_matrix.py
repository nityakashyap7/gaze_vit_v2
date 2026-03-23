import sklearn
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

confusion_matrices = []

def append_confusion_matrix(predictions, targets, epoch):

    matrix = sklearn.metrics.confusion_matrix(targets, predictions)
    plt.clf()
    img = sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    img.set_title('Confusion Matrix')
    img.set_xlabel('Predicted Labels')
    img.set_ylabel('True Labels')  
    img.text(10, 9, f"epoch {epoch}") #cooked sry
    fig = img.get_figure()
    fig.canvas.draw()
    ## idk what these next three lines do. claude locked in
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frame = frame[:, :, :3] 

    confusion_matrices.append(frame)

#append_confusion_matrix(predictions, targets, 1)

def create_video(fps=2):
    if not confusion_matrices:
        return
    h, w = confusion_matrices[0].shape[:2]
    writer = cv2.VideoWriter('confusion_matrices.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    # make it from rgb to bgr
    for matrix in confusion_matrices:
        switched_matrix = cv2.cvtColor(matrix, cv2.COLOR_RGB2BGR)
        writer.write(switched_matrix)
    writer.release()








