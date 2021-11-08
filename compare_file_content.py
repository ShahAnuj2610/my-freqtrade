import requests


# get github file content
def get_github_file_content(file_path):
    repo_url = 'https://raw.githubusercontent.com/'
    repo_url += 'ShahAnuj2610/my-freqtrade/master/'
    repo_url += file_path
    return requests.get(repo_url).text


# compare github file content with local file content
def compare_file_content(file_path):
    local_file_content = get_local_file_content(file_path)
    github_file_content = get_github_file_content(file_path)
    if local_file_content == github_file_content:
        print('{} is up to date'.format(file_path))
    else:
        print('{} is not up to date'.format(file_path))
        print('local content:')
        print(local_file_content)
        print('github content:')
        print(github_file_content)
        # copy github    file content to local file
        with open(file_path, 'w') as f:
            f.write(github_file_content)
        # execute docker-compose        up
        print('docker-compose up')
        import subprocess
        subprocess.call(['docker-compose', 'up', '-d', '--force-recreate'])


# get local file content
def get_local_file_content(file_path):
    with open(file_path, 'r') as f:
        return f.read()


if __name__ == '__main__':
    # run compare_file_content every 15 minutes
    while True:
        compare_file_content('user_data/pairlist-volume-binance-usdt.json')
        compare_file_content('user_data/blacklist-binance.json')
        compare_file_content('user_data/config.live.json')

        # sleep for 15 minutes
        import time

        time.sleep(60 * 15)
