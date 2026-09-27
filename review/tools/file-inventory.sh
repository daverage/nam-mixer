#!/bin/bash
echo "=== Python Source Files ===" > review/tools/source-inventory.txt
find {hybrid,routes,scripts,tests,cloud} -maxdepth 10 -type f -name "*.py" 2>/dev/null | while read f; do
  lines=$(wc -l < "$f")
  commits=$(git log --oneline "$f" 2>/dev/null | wc -l)
  echo "$lines lines | $commits commits | $f" >> review/tools/source-inventory.txt
done | sort -rn

echo "" >> review/tools/source-inventory.txt
echo "=== JavaScript Files ===" >> review/tools/source-inventory.txt
find static -maxdepth 10 -type f \( -name "*.js" -o -name "*.jsx" -o -name "*.ts" -o -name "*.tsx" \) 2>/dev/null | while read f; do
  lines=$(wc -l < "$f")
  commits=$(git log --oneline "$f" 2>/dev/null | wc -l)
  echo "$lines lines | $commits commits | $f" >> review/tools/source-inventory.txt
done | sort -rn

echo "" >> review/tools/source-inventory.txt
echo "=== Config/Manifest Files ===" >> review/tools/source-inventory.txt
find . -maxdepth 1 -type f \( -name "*.txt" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.cfg" -o -name "*.ini" \) ! -path "./.git/*" 2>/dev/null | while read f; do
  lines=$(wc -l < "$f" 2>/dev/null || echo 0)
  echo "$lines lines | $f" >> review/tools/source-inventory.txt
done | sort -rn
