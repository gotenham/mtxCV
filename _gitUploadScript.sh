#!/bin/bash

# Check if .git directory exists at current location
if [ ! -d ".git" ]; then
    echo "Error: Git repository not initialised in the current directory. Run 'git init' and 'git remote add origin {GitRepoURL}' first."
    exit 1
fi

# Get initials of the executing user
user_initials=$(whoami | awk '{print toupper(substr($1,1,1))}')

# Add all files in the current folder
git add .

# Commit changes with a timestamp and user initials in the commit message
commit_message="Auto commit by [$user_initials] at [$(date +'%Y-%m-%d %H:%M:%S')]"
git commit -m "$commit_message"

# Push changes to the specified remote repository
remote_repo="origin" # << Update accordingly
branch="main"  # << Update accordingly

git push -u $remote_repo $branch

echo "-------------"
echo ">>Attempted auto commit and push to [$remote_repo] on branch [$branch] with commit message '$commit_message'."
read -n1 -r -p "Press any key to continue..." key