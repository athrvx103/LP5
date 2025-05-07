// Code to implement BFS using OpenMP: 
#include <iostream>
#include <queue>
#include <omp.h>
using namespace std;

class Node {
public:
    int data;
    Node* left;
    Node* right;
};

/*class BreadthFS {
public:
    Node* insert(Node* root, int data);
    void bfs(Node* root);
};

// Function to insert a node in level-order
Node* BreadthFS::insert(Node* root, int data) {
    Node* newNode = new Node();
    newNode->data = data;
    newNode->left = newNode->right = nullptr;

    if (!root)
        return newNode;

    queue<Node*> q;
    q.push(root);

    while (!q.empty()) {
        Node* temp = q.front();
        q.pop();

        if (!temp->left) {
            temp->left = newNode;
            return root;
        } else {
            q.push(temp->left);
        }

        if (!temp->right) {
            temp->right = newNode;
            return root;
        } else {
            q.push(temp->right);
        }
    }

    return root;
}

// Parallel BFS using OpenMP
void BreadthFS::bfs(Node* root) {
    if (!root)
        return;

    queue<Node*> q;
    q.push(root);

    cout << "BFS traversal:\n";

    while (!q.empty()) {
        int qSize = q.size();

        // Parallel loop over current level
        #pragma omp parallel for
        for (int i = 0; i < qSize; i++) {
            Node* currNode;

            // Critical section for queue access
            #pragma omp critical
            {
                if (!q.empty()) {
                    currNode = q.front();
                    q.pop();
                    cout << "\t" << currNode->data;

                    if (currNode->left)
                        q.push(currNode->left);
                    if (currNode->right)
                        q.push(currNode->right);
                }
            }
        }
    }

    cout << endl;
}

int main() {
    Node* root = nullptr;
    BreadthFS bfsTree;
    int data;
    char ans;

    do {
        cout << "Enter data => ";
        cin >> data;
        root = bfsTree.insert(root, data);
        cout << "Do you want to insert one more node? (y/n): ";
        cin >> ans;
    } while (ans == 'y' || ans == 'Y');

    bfsTree.bfs(root);

    return 0;
}*/






// Code to implement DFS using OpenMP: 
#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
using namespace std;

const int MAX = 100000;
vector<int> graph[MAX];
bool visited[MAX];

void dfs(int start) {
    stack<int> s;
    s.push(start);

    while (!s.empty()) {
        int curr_node;

        #pragma omp critical
        {
            curr_node = s.top();
            s.pop();
        }

        if (!visited[curr_node]) {
            #pragma omp critical
            visited[curr_node] = true;

            cout << curr_node << " ";

            #pragma omp parallel for
            for (int i = 0; i < graph[curr_node].size(); i++) {
                int adj_node = graph[curr_node][i];
                if (!visited[adj_node]) {
                    #pragma omp critical
                    s.push(adj_node);
                }
            }
        }
    }
}

int main() {
    int n, m, start_node;
    cout << "Enter number of nodes, edges and the start node: ";
    cin >> n >> m >> start_node;

    cout << "Enter pairs of edges:\n";
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u); // For undirected graph
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    cout << "DFS traversal:\n";
    dfs(start_node);

    return 0;
}
