## 2026-02-02 - Optimize Face Recognition with In-Memory Embeddings
**Learning:** DeepFace's `find` method scans the filesystem which is slow (O(N) file reads). Database storage of embeddings allows for O(1) read (fetch all) and fast in-memory vectorized comparison (O(N) in memory).
**Action:** When working with ML models that extract features (embeddings), always store the features and perform similarity search in memory (or vector DB) rather than re-computing or reading files for every request.
