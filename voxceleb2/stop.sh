pattern="whisper"

# List screen sessions, grep for the pattern, extract the session IDs, and kill each session
screen -ls | grep "$pattern" | awk -F '.' '{print $1}' | while read session_id; do
    screen -S "$session_id" -X quit
    echo "Killed screen session: $session_id"
done