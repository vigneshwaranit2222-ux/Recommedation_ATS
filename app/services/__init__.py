"""Service layer package.

Services encapsulate business logic and external I/O (Hugging Face router,
ChromaDB, scikit-learn scoring) so that routers stay thin — routers do
only validation, I/O orchestration, and error translation.
"""