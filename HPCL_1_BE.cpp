#include <iostream>
#include <stack>
#include <queue>
#include <omp.h>
using namespace std;

class TreeNode {
public:
    int data;
    TreeNode *left, *right;

    TreeNode(int value) {
        data = value;
        left = right = nullptr;
    }
};

class Tree {
public:
    TreeNode *root;

    Tree() { root = nullptr; }

    void insert(int value) {
        TreeNode *newNode = new TreeNode(value);
        if (!root) {
            root = newNode;
            return;
        }

        queue<TreeNode *> q;
        q.push(root);

        while (!q.empty()) {
            TreeNode *current = q.front();
            q.pop();
            if (!current->left) {
                current->left = newNode;
                return;
            } else {
                q.push(current->left);
            }

            if (!current->right) {
                current->right = newNode;
                return;
            } else {
                q.push(current->right);
            }
        }
    }

    void bfs() {
        if (!root) return;

        queue<TreeNode *> q;
        q.push(root);

        cout << "BFS: ";
        while (!q.empty()) {
            int n = q.size();
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                TreeNode *current;

                #pragma omp critical
                {
                    current = q.front();
                    q.pop();
                }
                cout << current->data << " ";

                if (current->left) {
                    #pragma omp critical
                    q.push(current->left);
                }
                if (current->right) {
                    #pragma omp critical
                    q.push(current->right);
                }
            }
        }
        cout << endl;
    }

    void dfs() {
        if (!root) return;

        stack<TreeNode *> s;
        s.push(root);

        cout << "DFS: ";
        while (!s.empty()) {
            int n = s.size();
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                TreeNode *current;

                #pragma omp critical
                {
                    current = s.top();
                    s.pop();
                }

                cout << current->data << " ";

                if (current->right) {
                    #pragma omp critical
                    s.push(current->right);
                }
                if (current->left) {
                    #pragma omp critical
                    s.push(current->left);
                }
            }
        }
        cout << endl;
    }
};

int main() {
    Tree tree;
    while (true) {
        int choice;
        cout << "Enter 1-Insert | 2-BFS | 3-DFS | 0-Exit: ";
        cin >> choice;

        if (choice == 1) {
            int value;
            cout << "Enter a number: ";
            cin >> value;
            tree.insert(value);
        } else if (choice == 2) {
            tree.bfs();
        } else if (choice == 3) {
            tree.dfs();
        } else if (choice == 0) {
            cout << "Exited successfully." << endl;
            break;
        } else {
            cout << "Invalid input. Try again." << endl;
        }
    }
    return 0;
}
