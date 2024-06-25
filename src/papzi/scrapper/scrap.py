import concurrent.futures


from papzi.constants import BASE_DIR
from papzi.scrapper.scrapper import run_scrapper


def split_list(input_list, num_jobs):
    # Calculate the length of each chunk
    chunk_size = len(input_list) // num_jobs
    remainder = len(input_list) % num_jobs

    divided_list = []
    start = 0
    for i in range(num_jobs):
        end = start + chunk_size + (1 if i < remainder else 0)
        divided_list.append(input_list[start:end])
        start = end

    return divided_list


def main():
    with open(BASE_DIR / "celebs.txt") as f:
        celeb_names = f.read().split("\n")
    # celeb_names = [i.name for i in (BASE_DIR / "a" / "validation").iterdir()]
    scrapped_dir = BASE_DIR / "scraped"
    if scrapped_dir.exists():
        already_fetched = [i.name for i in scrapped_dir.iterdir()]
        celeb_names = list(set(celeb_names) - set(already_fetched))
    else:
        scrapped_dir.mkdir()

    print(len(celeb_names))

    with concurrent.futures.ProcessPoolExecutor() as worker:
        futures = []
        for per_worker_celebs in split_list(celeb_names, 5):

            futures.append(worker.submit(run_scrapper, per_worker_celebs))

        for future in concurrent.futures.as_completed(futures):
            future.result()  # To catch exceptions if any


if __name__ == "__main__":
    main()
