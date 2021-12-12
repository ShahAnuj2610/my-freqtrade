# a script that deletes and regenerates the docker image

def main():
    import os
    import sys
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
    subprocess.call(["docker-compose", "up", "-d", "--force-recreate"])


if __name__ == "__main__":
    main()
