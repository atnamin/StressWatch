mkdir -p ~/.streamlit
echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamli/config.toml 