# validation_Math_20

- 과목: Math / 유형: multiple-choice
- **정답: B**
- 모델 최종 답 (seed 3407 / 1 / 2): A / A / A
- seed 3407 추출 줄: `A`

## 이미지

**image 1** (510×179 px)

![image 1](../images/validation_Math_20_1.png)

## 문제

Find the minimum spanning tree of the graph G(V, U) of figure 1. Notice that it is an undirected graph.
<image 1>

- **A.** Graph G1 with vertices v1 to v8 and arcs including (v1, v3), (v2, v3) and so on.
- **B.** Graph G2 with vertices v1 to v8 and arcs including (v2, v5) and (v4, v8).
- **C.** Graph G1 with vertices v1 to v8 and arcs excluding (v2, v5) and (v4, v8).
- **D.** Graph G2 with vertices v1 to v8 and no arc connecting A1 and A3.

## 모델 응답 (seed 3407 전문)

```text
To find the minimum spanning tree (MST) of an undirected graph, we need to select a subset of edges that connects all vertices without forming any cycles, and with the minimum total edge weight.

The graph has 8 vertices: labeled 1 through 8. We are given edge weights between them.

Let’s list all edges and their weights from the diagram:

Edges:
- (1,2): 6
- (1,3): 5
- (1,4): 14
- (1,8): 18
- (2,3): 2
- (2,5): 8
- (3,4): 11
- (3,5): 16
- (3,6): 7
- (4,5): 4
- (4,6): 15
- (4,8): 9
- (5,6): 4
- (5,7): 17
- (6,7): 12
- (6,8): 13
- (7,8): 3

Note: The diagram shows some edges with multiple labels — for example, between 1 and 3 it says "5", but also "18" and "14" — this likely means there are multiple edges? But in standard MST problems, unless specified otherwise, we assume simple undirected graph with one edge per pair. Looking at the diagram again, it appears that each pair of vertices is connected by exactly one edge with a single weight label. So we’ll take the single weight shown for each edge.

We can use Kruskal’s algorithm (or Prim’s) to find the MST.

**Step 1: Sort all edges by weight**

Sorted edges (ascending):

1. (2,3): 2
2. (1,3): 5
3. (4,5): 4
4. (5,6): 4
5. (3,6): 7
6. (7,8): 3 ← Wait! This is 3, which is smaller than 4.

Wait — I missed that. Let me re-sort properly.

List of edges with weights:

- (7,8): 3
- (2,3): 2
- (4,5): 4
- (5,6): 4
- (1,3): 5
- (3,6): 7
- (4,6): 15
- (4,8): 9
- (1,4): 14
- (1,8): 18
- (2,5): 8
- (3,4): 11
- (3,5): 16
- (5,7): 17
- (6,7): 12
- (6,8): 13
- (1,2): 6

Now sort by weight:

1. (7,8): 3
2. (2,3): 2
3. (4,5): 4
4. (5,6): 4
5. (1,3): 5
6. (3,6): 7
7. (4,6): 15
8. (4,8): 9
9. (1,4): 14
10. (1,8): 18
11. (2,5): 8
12. (3,4): 11
13. (3,5): 16
14. (5,7): 17
15. (6,7): 12
16. (6,8): 13
17. (1,2): 6

Wait — I think I made a mistake: (1,2) is labeled "6", so it should be included. And (2,5) is 8, etc.

But (7,8) is 3, (2,3) is 2 — these are the smallest.

So sorted correctly:

1. (2,3): 2
2. (7,8): 3
3. (4,5): 4
4. (5,6): 4
5. (1,3): 5
6. (1,2): 6
7. (3,6): 7
8. (4,6): 15
9. (4,8): 9
10. (1,4): 14
11. (1,8): 18
12. (2,5): 8
13. (3,4): 11
14. (3,5): 16
15. (5,7): 17
16. (6,7): 12
17. (6,8): 13

Actually, let's double-check (1,2): Is it labeled 6? Yes, on the diagram, between 1 and 2, it says "6". Similarly, (2,5) is labeled "8".

So now apply Kruskal’s algorithm:

Start with empty set.

Add (2,3): weight 2 → connects {2,3}

Add (7,8): weight 3 → connects {7,8}

Add (4,5): weight 4 → connects {4,5}

Add (5,6): weight 4 → connects {4,5,6} — no cycle

Add (1,3): weight 5 → connects {1,3} — now {1,2,3,4,5,6,7,8}? Not yet — 1 is not connected to others. Adding (1,3) connects 1 to {2,3}, so now 1 is connected.

Current components:

- {1,2,3}
- {4,5,6}
- {7,8}

Next edge: (1,2): weight 6 — connects 1 and 2 — already connected, skip.

Next: (3,6): weight 7 — connects {1,2,3} and {4,5,6} — now they’re connected. So now we have {1,2,3,4,5,6} and {7,8}

Next: (4,6): weight 15 — both in same component, skip.

Next: (4,8): weight 9 — connects {1,2,3,4,5,6} and {7,8} — now all 8 vertices connected.

Total weight = 2 + 3 + 4 + 4 + 5 + 7 + 9 = 34

We have 7 edges (for 8 vertices), which is correct for MST.

The edges chosen are:

- (2,3): 2
- (7,8): 3
- (4,5): 4
- (5,6): 4
- (1,3): 5
- (3,6): 7
- (4,8): 9

Now, look at the options.

Option A: Graph G1 with vertices v1 to v8 and arcs including (v1,v3), (v2,v3) and so on.

→ (v1,v3) is edge (1,3) — yes, included.
→ (v2,v3) is (2,3) — yes, included.

Also, (v4,v5) is (4,5) — yes.
(v5,v6) is (5,6) — yes.
(v3,v6) is (3,6) — yes.
(v4,v8) is (4,8) — yes.

So Option A describes exactly the MST we found.

Option B: Graph G2 with vertices v1 to v8 and arcs including (v2,v5) and (v4,v8).

→ (v2,v5) is (2,5) — weight 8 — NOT selected in our MST. We skipped it because it would create a cycle or connect components unnecessarily. In fact, (2,5) connects {2,3} to {4,5,6} — but we already connected those via (1,3) and (3,6). So adding (2,5) would be redundant and increase cost.

Also, (v4,v8) is included — yes, but only half of option B’s description.

Option C: Graph G1 with vertices v1 to v8 and arcs excluding (v2,v5) and (v4,v8).

→ Excludes (v4,v8) — but we DID include it. So wrong.

Option D: Graph G2 with vertices v1 to v8 and no arc connecting A1 and A3.

→ “A1 and A3” — probably refers to vertices 1 and 3. We DO have edge (1,3) in MST. So excluding it is wrong.

Therefore, only Option A matches the MST we constructed.

Answer: A
```
