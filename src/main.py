import os
from windbg_interface import WindbgInterface, load_config

def main():
    print("🚀 Windbg Analysis Tool - Multi-Command Output Capture 🚀")
    cdb_path, minidump_path = load_config()
    if not cdb_path or not minidump_path:
        print("❌ Error: CDB_PATH or MINIDUMP_PATH not set in .env file. ❌")
        return
    
    debugger = WindbgInterface(cdb_path, minidump_path)

    minidump_filename = os.path.basename(minidump_path)
    minidump_date = minidump_filename.split('-')[0] if '-' in minidump_filename else "unknown_date"

    base_output_dir = os.path.join("command_outputs", minidump_date)
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"✅ Created output directory: {base_output_dir}")
    
    commands_to_run = [
        "kv",
        "!thread",
        "!devstack"
    ]

    print("\n--- ⏳ Starting Multi-Command Execution ⏳ ---")
    for command in commands_to_run:
        print(f"\n▶️ Executing command: '{command}'")
        sanitized_command_name = ''.join(c if c.isalnum() else '_' for c in command)
        output_file_name = f"{sanitized_command_name}.txt"
        output_file_path = os.path.join(base_output_dir, output_file_name)

        parsed_output = debugger.execute_command_and_parse_output(command, timeout=90)

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(parsed_output)
            
        print(f"💾 Parsed output for '{command}' written to {output_file_path}")
    
    print("\n--- 🎉 All Commands Executed and Outputs Saved 🎉 ---")

if __name__ == "__main__":
    main()
