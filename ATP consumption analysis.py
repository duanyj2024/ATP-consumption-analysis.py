import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import dunnett
import matplotlib.pyplot as plt

t = np.array([0, 0.6253, 1.250717, 1.876183, 2.501733, 3.127133, 3.75265,
              4.37825, 5.003683, 5.62905, 6.254567, 6.880033, 7.50545,
              8.131067, 8.756433, 9.382017, 10.0078])

data = {
    "C2": [
        [5.531528824,6.09366254,6.631128155,7.188776761,7.702321792,8.253990252,8.656902583,9.30799097,9.574854981,10.02486096,10.28798738,10.59521738,10.95477365,11.22238518,11.37562642,11.74191036,11.84357284],
        [4.524621756,4.830356716,5.316243571,5.725136048,6.048811446,6.41659042,6.729053044,7.239608001,7.604396902,7.67765369,8.094768867,8.332479668,8.644942292,8.689793386,9.08298798,9.207823526,9.429836443],
        [5.447806781,5.750551668,6.094410059,6.543668521,6.943590779,7.256053403,7.719514711,7.961710621,8.263707989,8.770525356,9.03215674,9.181660388,9.455252063,9.614473448,9.752016804,10.07643972,10.29022994],
        [7.833137484,8.393028645,9.020196448,9.591300383,10.15567665,10.48084709,11.03924321,11.54381802,11.81591466,12.34067247,12.87589553,12.75105998,13.29899085,13.59276552,13.83720398,14.02333602,14.19078011],
    ],
    "C2+WT": [
        [4.08134344,4.230847088,4.470052924,4.707016206,4.987335546,5.162254814,5.340164155,5.492657876,5.861931886,5.858194295,6.227468305,6.343333632,6.521990492,6.634118227,6.631128155,6.830715524,7.022080194],
        [3.832419866,4.098536359,4.163570446,4.359420225,4.573210441,4.658427521,4.841569489,5.047884523,5.213833573,5.311010944,5.480697584,5.663839553,5.689255173,6.057781665,6.024143344,6.28502721,6.368749252],
        [3.912404318,4.052190228,4.153852709,4.434919567,4.454355041,4.68832825,4.941736933,4.952202189,5.182437806,5.533023861,5.680284954,5.768492106,5.95387663,5.944158892,6.05553911,6.231205896,6.286522246],
        [4.324286868,4.430434458,4.700288542,4.983597955,4.997800801,5.324466272,5.479202548,5.62272605,5.850719113,6.033113563,6.218498086,6.374729398,6.47190677,6.738023263,6.82398786,6.746993482,6.890516984],
    ],
    "C2+mut": [
        [5.812595682,6.302967647,7.017595084,7.596174202,7.984883686,8.558230176,8.849014771,9.491880457,9.972534685,10.39039738,10.75967139,10.93758073,11.46233854,11.74041532,11.94374028,11.99980415,12.34291502],
        [5.598057948,6.006950425,6.635613264,6.99815961,7.664198361,7.799499163,8.351167623,8.621021708,9.096443308,9.29528316,9.754259359,10.20725541,10.47411942,10.69986993,10.89571971,11.13642058,11.43169029],
        [6.180374656,6.427803193,6.936115596,7.416769824,7.822672228,8.319771857,8.784728202,8.887138201,9.368539947,9.886570087,10.21547811,10.5675592,10.80452249,11.09754964,11.43468036,11.6574408,11.82936999],
        [7.951992884,8.688298349,9.374520093,9.877599868,10.76565154,11.52139248,12.24050502,12.84076217,13.21825888,13.69293296,14.05996442,14.59892507,14.82019047,15.38157667,15.55275834,16.16796585,16.1657233],
    ],
}

print("R2 scan (full 0-10min window), per replicate:")
for group, reps in data.items():
    print(f"\n--- {group} ---")
    for i, y in enumerate(reps):
        y = np.array(y)
        slope, intercept, r, p, se = stats.linregress(t, y)
        print(f"  rep{i+1}: slope={slope:.5f} uM/min, R2={r**2:.4f}")

records = []
for group, reps in data.items():
    for i, y in enumerate(reps):
        y = np.array(y)
        slope, intercept, r, p, se = stats.linregress(t, y)
        records.append({"Group": group, "Replicate": i+1,
                         "Rate_uM_ATP_per_min": slope, "R2": r**2})

df = pd.DataFrame(records)
summary = df.groupby("Group")["Rate_uM_ATP_per_min"].agg(["mean","std","sem","count"])
summary = summary.reindex(["C2","C2+WT","C2+mut"])
print("\nSummary (uM ATP/min):")
print(summary)

g_C2 = df[df.Group=="C2"].Rate_uM_ATP_per_min.values
g_mut = df[df.Group=="C2+mut"].Rate_uM_ATP_per_min.values
g_WT = df[df.Group=="C2+WT"].Rate_uM_ATP_per_min.values
result = dunnett(g_C2, g_mut, control=g_WT)
print("\nDunnett's test (control = C2+WT):")
print(result)
print(f"  C2 vs C2+WT:     p = {result.pvalue[0]:.6f}")
print(f"  C2+mut vs C2+WT: p = {result.pvalue[1]:.6f}")

groups_order = ["C2","C2+WT","C2+mut"]
means = [summary.loc[g,"mean"] for g in groups_order]
sems  = [summary.loc[g,"sem"] for g in groups_order]
colors = ['white','#e74c3c','#2962ff']

fig, ax = plt.subplots(figsize=(4.5,5))
x = np.arange(len(groups_order))
ax.bar(x, means, yerr=sems, capsize=5, width=0.6, color=colors, edgecolor='black', linewidth=1.5)
rng = np.random.default_rng(0)
for i,g in enumerate(groups_order):
    yvals = df[df.Group==g].Rate_uM_ATP_per_min.values
    xvals = rng.normal(i,0.05,size=len(yvals))
    ax.scatter(xvals, yvals, color='black', s=30, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(groups_order)
ax.set_ylabel("Initial rate (uM ATP / min)")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/rate_v5_barplot.png", dpi=300)
df.to_csv("/mnt/user-data/outputs/rate_v5.csv", index=False)

with pd.ExcelWriter("/mnt/user-data/outputs/rate_v5_final.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Per-replicate rates", index=False)
    summary.to_excel(writer, sheet_name="Summary")
print("\nSaved outputs.")
