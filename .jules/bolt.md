## 2024-05-23 - [In-Memory Embedding Comparison]
**Learning:** The application was performing face recognition by scanning the file system and using `DeepFace.find` or `DeepFace.verify` on every request. This scales linearly with the number of images (O(N)) and involves heavy disk I/O.
**Action:** Implemented in-memory vector comparison using pre-calculated embeddings stored in the database. This eliminates disk I/O during recognition and allows for much faster O(N) comparisons (or O(log N) with vector indexes in the future).
