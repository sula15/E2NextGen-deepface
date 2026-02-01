## 2024-05-23 - Vectorized Face Recognition
**Learning:** `DeepFace.find` relies on file system scanning and on-disk pickle caches which is slow (O(N) or worse with disk I/O) and fragile (cache invalidation).
**Action:** Store embeddings in the database (JSON column) and use vectorized in-memory comparison (NumPy) for O(1) inference + O(N) fast vector math. Use `User.query.with_entities(User.embedding)` to fetch only necessary data.
