#include <iostream>
#include <vector>
#include <queue>
#include <omp.h> // OpenMP for parallelization
#include <chrono> // Chrono for timing

using namespace std;

// Graph class representing the adjacency list
class Graph {
public:
    int V;  // Number of vertices
    vector<vector<int>> adj;  // Adjacency list

    // Constructor to initialize the graph with given number of vertices
    Graph(int V) : V(V), adj(V) {}

    // Add an edge to the graph
    void addEdge(int v, int w) {
        adj[v].push_back(w);
    }

    // Parallel Depth-First Search with timing (using chrono)
    void parallelDFS(int startVertex) {
        vector<bool> visited(V, false);
        auto start_time = chrono::high_resolution_clock::now();
        auto end_time = chrono::high_resolution_clock::now();

        start_time = chrono::high_resolution_clock::now();
        parallelDFSUtil(startVertex, visited); // Call parallel DFS
        end_time = chrono::high_resolution_clock::now();

        // Calculate execution time
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

        // Print DFS traversal and execution time
        cout << "Depth-First Search (DFS): ";
        cout << endl << "Execution Time: " << duration.count() << " microseconds" << endl;
    }

    // Parallel DFS utility function
    void parallelDFSUtil(int sv, vector<bool>& visited) {
        visited[sv] = true;
        cout << sv << " "; // Print the current vertex

        // Parallelize DFS for each adjacent vertex using OpenMP
        #pragma omp parallel for
        for (int i = 0; i < adj[sv].size(); ++i) {
            int n = adj[sv][i];
            if (!visited[n])
                parallelDFSUtil(n, visited); // Recursive call
        }
    }

    // Parallel Breadth-First Search with timing (using chrono)
    void parallelBFS(int startVertex) {
        vector<bool> visited(V, false);
        queue<int> q;
        auto start_time = chrono::high_resolution_clock::now();
        auto end_time = chrono::high_resolution_clock::now();

        start_time = chrono::high_resolution_clock::now();

        visited[startVertex] = true;
        q.push(startVertex);

        while (!q.empty()) {
            int v = q.front();
            q.pop();
            cout << v << " "; // Print the current vertex

            // Parallelize BFS for each adjacent vertex using OpenMP
            #pragma omp parallel for
            for (int i = 0; i < adj[v].size(); ++i) {
                int n = adj[v][i];
                if (!visited[n]) {
                    visited[n] = true;
                    q.push(n);
                }
            }
        }

        end_time = chrono::high_resolution_clock::now();
        // Calculate execution time
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

        // Print BFS traversal and execution time
        cout << endl << "Breadth-First Search (BFS): ";
        cout << endl << "Execution Time: " << duration.count() << " microseconds" << endl;
    }
};

int main() {
    // Create a graph with 7 vertices
    Graph g(7);
    // Add edges to the graph
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);

    // Perform parallel DFS and BFS starting from vertex 0
    g.parallelDFS(0);
    g.parallelBFS(0);

    return 0;
}
