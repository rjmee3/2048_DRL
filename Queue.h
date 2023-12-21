#ifndef QUEUE_H
#define QUEUE_H

#define MAX_SIZE 4

typedef struct {
    int data[MAX_SIZE];
    int front;
    int rear;
} Queue;

/*  Initializes an empty queue.     */
void initializeQueue(Queue *queue);
/*  Returns 1 if queue is empty, 0 if otherwise.    */
int isEmpty(Queue *queue);
/*  Returns 1 if queue is full, 0 if otherwise.     */
int isFull(Queue *queue);
/*  Enqueues the element to the end of the queue.   */
void enqueue(Queue *queue, int element);
/*  Dequeues the element at the front of the queue.     */
int dequeue(Queue *queue);
/*  Returns the value of the element at the front of the queue.     */
int front(Queue *queue);

#endif