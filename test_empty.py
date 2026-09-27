from risk_logic import RiskAnalyzer

analyzer = RiskAnalyzer()

result = analyzer.process_file(
    "output/empty_events.json",
    "output/empty_risk_events.json"
)

print("Processed events:", len(result))
print("No animal alerts generated.")