import time

from tqdm import tqdm

from config.settings import settings
from repositories.file_repository import FileRepository
from services.ingestion.ingestion_processor import IngestionProcessor


def main():
    print("1. 开始扫描待入库的 markdown 文档")
    file_repository = FileRepository()
    file_paths = file_repository.list_files(settings.CRAWL_OUTPUT_DIR)

    print(f"2. 扫描得到文件数: {len(file_paths)}")
    unique_file_paths = file_repository.remove_duplicate_files(file_paths)
    print(f"3. 去重后文件数: {len(unique_file_paths)}")

    ingestion_processor = IngestionProcessor()
    success = 0
    fail = 0
    start_time = time.time()

    with tqdm(unique_file_paths, desc="知识库入库中") as pbar:
        for file_path in pbar:
            try:
                ingestion_processor.ingest_file(file_path, refresh_bm25=False)
                success += 1
            except Exception:
                fail += 1
            finally:
                pbar.set_postfix({"success": success, "fail": fail})

    if success:
        ingestion_processor.rebuild_bm25_index()

    total_time = time.time() - start_time
    print(f"4. 入库完成: success={success}, fail={fail}")
    print(f"5. 总耗时: {total_time:.2f}s")


if __name__ == "__main__":
    main()
