#!/bin/bash
#
if [ $# -eq 0 ]; then
    echo "No question parameter provided."
    exit 1
fi

cd ./graphrag-10k/ || exit

graphrag query --root . --method "$2" --query "$1"

#This is for testing purposes - use the line above:
#echo "Local Search Response:
### Apple Inc.'s Earnings in 2023
#
#In fiscal year 2023, Apple Inc. reported impressive financial results, showcasing its strong market position and operational efficiency. The company achieved total net sales of **$383.3 billion**, which reflects a **3% decrease** compared to the previous fiscal year. Despite this decline, Apple maintained a robust net income of **$97.0 billion**, indicating its ability to generate substantial profits even amidst challenging market conditions [Data: Reports (0); Entities (138)].
#
#### Breakdown of Financial Performance
#
#The decrease in net sales was primarily attributed to lower sales of key products, particularly the iPhone and Mac, which were partially offset by an increase in services revenue. Specifically, the services segment saw a **9% increase** in net sales, reaching **$85.2 billion** in 2023, highlighting Apple's successful diversification beyond hardware sales [Data: Reports (0); Sources (16)].
#
#### Conclusion
#
#Overall, Apple's financial performance in 2023 underscores its resilience and adaptability in a competitive technology landscape. The company's ability to maintain a high net income amidst a slight decline in total sales demonstrates its strong brand loyalty and effective business strategies [Data: Reports (0); Entities (138)].
#"
