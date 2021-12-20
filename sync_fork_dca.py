# check github repository for updates
# if there's an update, build and push docker image
# if there's no update, do nothing
import subprocess
import time


def check_for_github_repo_changes():
    repo_url = 'https://github.com/xataxxx/freqtrade'
    branch = 'dca'
    repo_path = 'freqtrade'

    # get the last commit hash
    last_commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_path).decode('utf-8').strip()

    # get the last commit hash from the remote
    remote_commit_hash = \
        subprocess.check_output(['git', 'ls-remote', '--heads', repo_url, branch], cwd=repo_path).decode('utf-8').split(
            '\t')[0]

    # compare the two
    if last_commit_hash != remote_commit_hash:
        # update the local repo
        subprocess.call(['git', 'fetch', '--all'], cwd=repo_path)
        subprocess.call(['git', 'checkout', '-f', branch], cwd=repo_path)
        subprocess.call(['git', 'reset', '--hard', remote_commit_hash], cwd=repo_path)
        return True
    else:
        return False


def build_and_push_docker_image():
    if check_for_github_repo_changes():
        print('changes found, building and pushing docker image')
        # build docker image with no cache
        subprocess.call(['docker', 'build', '-t', 'anujshah1996/freqtrade:latest', '--no-cache', '.'], cwd='freqtrade')
        subprocess.call(['docker', 'push', 'anujshah1996/freqtrade:latest'])
        # docker-compose up -d
        subprocess.call(['docker-compose', 'build'], cwd='my-freqtrade')
        subprocess.call(['docker-compose', 'up', '-d', '--force-recreate'], cwd='my-freqtrade')


if __name__ == '__main__':
    #     run this script every hour
    while True:
        build_and_push_docker_image()
        time.sleep(3600)
