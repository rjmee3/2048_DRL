#include "include/Queue.h"
#include <stdio.h>

void initializeQueue(Queue *queue) {
    // initial states indicate that queue has no elements
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
    // if queue is full, print err and return
    if (isFull(queue)) {
        fprintf(stderr, "Queue Full. Cannot Enqueue.\n");
        return;
    }

    // if queue is empty, init front to 0
    if (isEmpty(queue)){
        queue->front = 0;
    }

    // increment rear index, modulo is used to ensure 
    // index stays within bounds of array
    queue->rear = (queue->rear + 1) % MAX_SIZE;
    queue->data[queue->rear] = element;
}

int dequeue(Queue *queue) {
    // print error if queue is empty
    if (isEmpty(queue)) {
        fprintf(stderr, "Queue is Empty. Cannot Dequeue.\n");
        return -1;
    }

    // get element at front
    int element = queue->data[queue->front];

    // if last element is dequeued, set front and rear to initial states
    // else, increment front index
    if (queue->front == queue->rear) {
        queue->front = -1;
        queue->rear = -1;
    } else {
        queue->front = (queue->front+1) % MAX_SIZE;
    }

    return element;
}

int front(Queue *queue) {
    // print err if queue is empty
    if (isEmpty(queue)) {
        // fprintf(stderr, "Queue Empty. No Front Element.\n");
        return -1;
    }

    return queue->data[queue->front];
}