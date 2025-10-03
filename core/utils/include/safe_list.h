#pragma once

// 使用链表代替队列是否会更好些？（下一阶段任务）

struct ListNode {
    void* data;
    ListNode* next;
};

class safe_list {
    ListNode* list_head;
    ListNode* list_tail;
    int size;
public:
    void clear() {

    }
};