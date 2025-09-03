import os
import subprocess
from dotenv import load_dotenv

OUTPUT_START_DELIMITER = "---COMMAND_START---"
OUTPUT_END_DELIMITER = "---COMMAND_END---"

class WindbgInterface:
    def __init__(self, cdb_path, minidump_path):
        self.cdb_path = cdb_path
        self.minidump_path = minidump_path

    def _clean_analyze_v_output(self, output: str) -> str:
        """
        Clean up !analyze -v command output by removing decorative elements and noise.
        This removes:
        - Decorative asterisk header boxes (Bugcheck Analysis section)
        - "Debugging Details:" headers and separators
        - Lines that are just asterisks and spaces
        - Leading/trailing empty lines
        """
        if not output:
            return output
            
        lines = output.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip decorative asterisk headers (bugcheck analysis box)
            if line.strip().startswith('*') and line.strip().endswith('*'):
                continue
                
            # Skip lines that are just asterisks, spaces, or tabs
            if all(c in '*  \t' for c in line.strip()) and line.strip():
                continue
                
            # Skip the "Debugging Details:" header and its separator
            if line.strip() in ["Debugging Details:", "------------------", "---------"]:
                continue
                
            # Skip lines that are purely decorative (multiple dashes)
            if line.strip() and all(c in '-' for c in line.strip()):
                continue
                
            # Keep the line if it's not noise
            cleaned_lines.append(line)
        
        # Remove any leading/trailing empty lines
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
            
        return '\n'.join(cleaned_lines)

    def _is_analyze_v_command(self, command: str) -> bool:
        """
        Check if the command is a variant of !analyze -v.
        """
        cmd_lower = command.strip().lower().replace(' ', '').replace('-', '')
        return cmd_lower in ['!analyzev', '!analyze-v', '!analyze -v'.replace(' ', '').replace('-', '')]

    def execute_command_and_parse_output(self, command: str, timeout=60) -> str:
        cdb_cmd_string = f".echo {OUTPUT_START_DELIMITER}; {command}; .echo {OUTPUT_END_DELIMITER}; qq"
        
        cmd = [
            self.cdb_path,
            '-z', self.minidump_path,
            '-c', cdb_cmd_string
        ]

        print(f"▶️ Executing CDB command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
            full_output = result.stdout
            # print(f"📄 Full output from CDB:\n{full_output}")
            
            lines = full_output.splitlines()
            
            output_started = False
            extracted_lines = []
            
            passed_echo_line = False # Flag to skip the initial command echo from CDB

            for line in lines:
                if not passed_echo_line and "cdb: Reading initial command" in line:
                    passed_echo_line = True
                    continue # Skip this line as it's the debugger's echo

                if passed_echo_line and OUTPUT_START_DELIMITER in line and not output_started:
                    output_started = True
                    # Append content after the delimiter on the same line, if any
                    part_after_delimiter = line.split(OUTPUT_START_DELIMITER, 1)[-1].strip()
                    if part_after_delimiter:
                        extracted_lines.append(part_after_delimiter)
                    continue # Move to next line after finding start delimiter
                
                if OUTPUT_END_DELIMITER in line:
                    if output_started:
                        # Append content before the delimiter on the same line, if any
                        part_before_delimiter = line.split(OUTPUT_END_DELIMITER, 1)[0].strip()
                        if part_before_delimiter:
                            extracted_lines.append(part_before_delimiter)
                    output_started = False
                    break # Stop processing lines after end delimiter
                
                if output_started:
                    extracted_lines.append(line.strip())
            
            extracted_output = "\n".join(filter(None, extracted_lines)).strip()

            if extracted_output:
                # Apply special cleaning for !analyze -v command
                if self._is_analyze_v_command(command):
                    print("🧹 Applying special cleaning for !analyze -v command...")
                    original_line_count = len(extracted_output.splitlines())
                    extracted_output = self._clean_analyze_v_output(extracted_output)
                    cleaned_line_count = len(extracted_output.splitlines())
                    lines_removed = original_line_count - cleaned_line_count
                    print(f"✅ Cleaned !analyze -v Output ({lines_removed} noise lines removed):")
                    print(f"{extracted_output}")
                    print("-" * 50)
                else:
                    print(f"✅ Parsed Command Output (Terminal Display):\n{extracted_output}\n-------------------------------------------------")
                return extracted_output
            else:
                print(f"⚠️ Warning: No content extracted between delimiters.\nFull output:\n{full_output}")
                return full_output

        except subprocess.CalledProcessError as e:
            error_msg = f"❌ Error executing CDB command: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}"
            print(error_msg)
            return error_msg
        except subprocess.TimeoutExpired as e:
            error_msg = f"⏰ CDB command timed out after {timeout} seconds.\nStdout: {e.stdout}\nStderr: {e.stderr}"
            print(error_msg)
            return error_msg
        except FileNotFoundError:
            error_msg = f"🔍 Error: CDB.exe not found at {self.cdb_path}"
            print(error_msg)
            return error_msg
        except (OSError, RuntimeError) as e:
            error_msg = f"💥 An unexpected error occurred: {e}"
            print(error_msg)
            return error_msg

def load_config():
    load_dotenv()
    cdb_path = os.getenv('CDB_PATH')
    minidump_path = os.getenv('MINIDUMP_PATH')
    chonkie_api_key = os.getenv('CHONKIE_API_KEY')
    return cdb_path, minidump_path, chonkie_api_key
