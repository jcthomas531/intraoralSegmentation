#!/bin/bash
snakemake --rulegraph > fullRuleGraph.dot
grep -vE '^[[:space:]]*[0-9]+[[:space:]]*->[[:space:]]*0[[:space:]]*$' fullRuleGraph.dot > fullRuleGraphNice.dot
dot -Tpng fullRuleGraphNice.dot > fullRuleGraphNice.png
rm fullRuleGraphNice.dot
rm fullRuleGraph.dot
