#include "Queue.h"
#include <stdio.h>

void initializeQueue(Queue *queue) {
    queue->front = -1;
    queue->rear = -1;
}

int isEmpty(Queue *queue) {
    return (queue->front == -1 && queue->rear == -1);
}

int isFull(Queue *queue) {
    return ((queue->rear + 1) % MAX_SIZE == queue->front);
}

void enqueue(Queue *queue, int element) {
    
    if (isFull(queue)) {
        fprintf(stderr, "Queue Full. Cannot Enqueue.\n");
        return;
    }

    if (isEmpty(queue)){
        queue->front = 0;
    }

    queue->rear = (queue->rear + 1) % MAX_SIZE;
    queue->data[queue->rear] = element;
}

int dequeue(Queue *queue) {

}

int front(Queue *queue) {

}