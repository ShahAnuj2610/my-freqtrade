# a script that rebuilds the docker image whenever a new commit is pushed to the freqtrade develop branch


last_commit_id = ""


def regenerate_docker_image():
    import subprocess

    # find all docker images that have the name "freqtrade"
    docker_images = subprocess.check_output(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"]).decode().splitlines()
    for image in docker_images:
        if "freqtrade" in image:
            print(f"Deleting {image}")
            # force delete
            subprocess.call(["docker", "rmi", "-f", image])

    # run docker-compose to regenerate the image
    subprocess.call(["docker-compose", "up", "-d", "--force-recreate"], cwd='my-freqtrade')


def check_for_new_commit():
    global last_commit_id

    import subprocess

    repo_url = "https://github.com/freqtrade/freqtrade"
    branch = "develop"

    # get the latest commit id
    commit_id = subprocess.check_output(
        ["git", "ls-remote", repo_url, branch]).decode().splitlines()[0].split("\t")[0]

    if last_commit_id != commit_id:
        print(f"New commit detected: {commit_id}")
        last_commit_id = commit_id
        regenerate_docker_image()


if __name__ == "__main__":
    #     run the script every 15 minutes
    while True:
        check_for_new_commit()
        import time

        time.sleep(60 * 15)
