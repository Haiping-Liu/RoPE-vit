import torch
import torch.nn.functional as F
import h5py
import scipy
import argparse
import cv2
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
import scipy.stats
from tqdm import tqdm
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--load-pkl", action='store_true',
                    help="Load predictions from a cached pickle file or recompute from scratch")
args = parser.parse_args()

class Model(Enum):
    GroundTruth = "GroundTruth"
    Dropout = "Dropout"
    Ensemble = "Ensemble"
    Evidential = "Evidential"
    MAE = "MAE"

save_dir = "save/20250321-002233__Mae_"
trained_models = {
    # Model.Dropout: ["model_rmse_57500.pth"],
    # "Ensemble": ["ensemble/trial1_*.pt", "ensemble/trial2_*.pt", "ensemble/trial3_*.pt"],
    # "Evidential": ["evidence/trial1.pt", "evidence/trial2.pt", "evidence/trial3.pt"],
    Model.MAE: ["model_rmse_57000.pth"],
}
output_dir = "figs/depth"


def compute_predictions(batch_size=50, n_adv=9):
    (x_in, y_in), (x_ood, y_ood) = load_data()
    datasets = [(x_in, y_in, False), (x_ood, y_ood, True)]

    df_pred_image = pd.DataFrame(columns=["Method", "Model Path", "Input", "Target", "Mu", "Sigma", "Adv. Mask", "Epsilon", "OOD"])
    adv_eps = np.linspace(0, 0.04, n_adv)

    for method, model_path_list in trained_models.items():
        for model_path in model_path_list:
            full_path = os.path.join(save_dir, model_path)
            print(method.value)
            model = models.load_depth_model(method.value, full_path).to(device)
            model.eval()

            for x, y, ood in datasets:
                for start_i in tqdm(range(0, len(x), batch_size)):
                    inds = np.arange(start_i, min(start_i + batch_size, len(x)))

                    x_batch = torch.tensor(np.transpose(x[inds] / 255.0, (0, 3, 1, 2)), dtype=torch.float32,
                                           device=device)
                    y_batch = torch.tensor(np.transpose(y[inds] / 255.0, (0, 3, 1, 2)), dtype=torch.float32,
                                           device=device)

                    # img = x_batch[0].cpu().numpy()
                    # img = np.transpose(img, (1, 2, 0))
                    # plt.imshow(img)
                    # plt.axis("off")
                    # plt.show()

                    if ood:
                        summary_to_add = get_prediction_summary(method, model_path, model, x_batch, y_batch, ood)
                        df_pred_image = pd.concat([df_pred_image, pd.DataFrame(summary_to_add)], ignore_index=True)
                    else:
                        mask_batch = create_adversarial_pattern(model, x_batch, y_batch)

                        for eps in adv_eps:
                            x_adv = x_batch + (eps * mask_batch)
                            x_adv = torch.clamp(x_adv, 0, 1)

                            summary_to_add = get_prediction_summary(method, model_path, model, x_adv, y_batch, ood, mask_batch, eps)
                            df_pred_image = pd.concat([df_pred_image, pd.DataFrame(summary_to_add)], ignore_index=True)

    return df_pred_image


def get_prediction_summary(method, model_path, model, x_batch, y_batch, ood, mask_batch=None, eps=0.0):
    if mask_batch is None:
        mask_batch = torch.zeros_like(x_batch)

    mu_batch, sigma_batch = predict(method, model, x_batch)
    print(mu_batch.shape, sigma_batch.shape,y_batch.shape)
    print(mu_batch)
    print(y_batch)
    rmse = torch.sqrt(F.mse_loss(mu_batch, y_batch))
    print(rmse)

    fig, axs = plt.subplots(5, 1, figsize=(6, 20))  # 5行1列，图像高度设得高一点

    # 图像1：RGB图
    img = x_batch[0].detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    axs[0].imshow(img)
    axs[0].axis("off")
    axs[0].set_title("Input Image")

    # 图像2：GT深度图
    depth_map = y_batch[0, 0].detach().cpu().numpy()
    im = axs[1].imshow(depth_map, cmap='magma')
    axs[1].axis("off")
    axs[1].set_title("Depth Map GT")
    # fig.colorbar(im, ax=axs[1], shrink=0.7)

    # 图像3：预测深度图
    depth_map = mu_batch[0, 0].detach().cpu().numpy()
    im = axs[2].imshow(depth_map, cmap='magma')
    axs[2].axis("off")
    axs[2].set_title("Depth Map Pred")
    # fig.colorbar(im, ax=axs[2], shrink=0.7)

    # 图像4：误差图
    depth_map = abs(y_batch[0, 0] - mu_batch[0, 0]).detach().cpu().numpy()
    im = axs[3].imshow(depth_map, cmap='magma')
    axs[3].axis("off")
    axs[3].set_title("Error Map")
    # fig.colorbar(im, ax=axs[3], shrink=0.7)

    # 图像5：不确定性图
    depth_map = sigma_batch[0, 0].detach().cpu().numpy()
    im = axs[4].imshow(depth_map, cmap='magma')
    axs[4].axis("off")
    axs[4].set_title("Uncertainty Map")
    # fig.colorbar(im, ax=axs[4], shrink=0.7)

    plt.tight_layout()
    plt.show()

    mu_batch = torch.clamp(mu_batch, 0, 1).detach().cpu().numpy()
    sigma_batch = sigma_batch.detach().cpu().numpy()

    summary = [{"Method": method.value, "Model Path": model_path, "Input": x, "Target": y, "Mu": mu, "Sigma": sigma,
                "Adv. Mask": mask, "Epsilon": eps, "OOD": ood}
               for x, y, mu, sigma, mask in zip(x_batch.detach().cpu().numpy(), y_batch.detach().cpu().numpy(), mu_batch, sigma_batch, mask_batch.detach().cpu().numpy())]
    return summary


def load_data():
    def load_depth():
        train = h5py.File("data/depth_train.h5", "r")
        test = h5py.File("data/depth_test.h5", "r")
        return (train["image"], train["depth"]), (test["image"], test["depth"])

    def load_apollo():
        test = h5py.File("data/apolloscape_test.h5", "r")
        return (None, None), (test["image"], test["depth"])

    _, (x_test, y_test) = load_depth()
    _, (x_ood_test, y_ood_test) = load_apollo()
    print("Loaded data:", x_test.shape, x_ood_test.shape)
    return (x_test, y_test), (x_ood_test, y_ood_test)


def predict(method, model, x, n_samples=10):
    with torch.no_grad():
        if method == Model.Dropout:
            model.train()
            preds = torch.stack([model(x) for _ in range(n_samples)], dim=0)
            mu = preds.mean(dim=0)
            sigma = preds.std(dim=0)
            return mu, sigma

        elif method == Model.Evidential:
            outputs = model(x)
            mu, v, alpha, beta = torch.split(outputs, 4, dim=-1)
            sigma = torch.sqrt(beta / (v * (alpha - 1)))
            return mu, sigma

        elif method == Model.Ensemble:
            preds = torch.stack([model_i(x) for model_i in model], dim=0)
            mu = preds.mean(dim=0)
            sigma = preds.std(dim=0)
            return mu, sigma

        elif method == Model.MAE:
            pred, mae, mae_up, mae_down = model(x)
            return pred, mae

        else:
            raise ValueError("Unknown model")


# def create_adversarial_pattern(model, x, y):
#     x.requires_grad = True
#     model.zero_grad()
#     # pred = model(x)
#     pred,_,_,_ = model(x)
#     loss = F.mse_loss(pred, y)
#     loss.backward()
#     signed_grad = x.grad.sign()
#     return signed_grad

# def df_image_to_pixels(df, keys=["Target", "Mu", "Sigma"]):
#     required_keys = ["Method", "Model Path"]
#     keys = required_keys + keys
#     key_types = {key: type(df[key].iloc[0]) for key in keys}
#     max_shape = max([np.prod(np.shape(df[key].iloc[0])) for key in keys])

#     contents = {}
#     for key in keys:
#         if np.prod(np.shape(df[key].iloc[0])) == 1:
#             contents[key] = np.repeat(df[key], max_shape)
#         else:
#             contents[key] = np.stack(df[key], axis=0).flatten()

#     df_pixel = pd.DataFrame(contents)
#     return df_pixel

# def apply_cmap(gray, cmap=cv2.COLORMAP_MAGMA):
#     if gray.dtype == np.float32:
#         gray = np.clip(255*gray, 0, 255).astype(np.uint8)
#     im_color = cv2.applyColorMap(gray, cmap)
#     return im_color

# def trim(img, k=10):
#     return img[k:-k, k:-k]
# def normalize(x, t_min=0, t_max=1):
#     return ((x-x.min())/(x.max()-x.min())) * (t_max-t_min) + t_min


# def gen_cutoff_plot(df_image, eps=0.0, ood=False, plot=True):
#     print(f"Generating cutoff plot with eps={eps}, ood={ood}")

#     df = df_image[(df_image["Epsilon"]==eps) & (df_image["OOD"]==ood)]
#     df_pixel = df_image_to_pixels(df, keys=["Target", "Mu", "Sigma"])

#     df_cutoff = pd.DataFrame(
#         columns=["Method", "Model Path", "Percentile", "Error"])

#     for method, model_path_list in trained_models.items():
#         for model_i, model_path in enumerate(tqdm(model_path_list)):
#             print(method.value)

#             df_model = df_pixel[(df_pixel["Method"]==method.value) & (df_pixel["Model Path"]==model_path)]
#             df_model = df_model.sort_values("Sigma", ascending=False)
#             percentiles = np.arange(100)/100.
#             cutoff_inds = (percentiles * df_model.shape[0]).astype(int)

#             df_model["Error"] = np.abs(df_model["Mu"] - df_model["Target"])
#             mean_error = [df_model[cutoff:]["Error"].mean()
#                 for cutoff in cutoff_inds]
#             df_single_cutoff = pd.DataFrame({'Method': method.value, 'Model Path': model_path,
#                 'Percentile': percentiles, 'Error': mean_error})

#             df_cutoff = df_cutoff.append(df_single_cutoff)

#     df_cutoff["Epsilon"] = eps

#     if plot:
#         print("Plotting cutoffs")
#         sns.lineplot(x="Percentile", y="Error", hue="Method", data=df_cutoff)
#         plt.savefig(os.path.join(output_dir, f"cutoff_eps-{eps}_ood-{ood}.pdf"))
#         plt.show()

#         sns.lineplot(x="Percentile", y="Error", hue="Model Path", style="Method", data=df_cutoff)
#         plt.savefig(os.path.join(output_dir, f"cutoff_eps-{eps}_ood-{ood}_trial.pdf"))
#         plt.show()

#         g = sns.FacetGrid(df_cutoff, col="Method", legend_out=False)
#         g = g.map_dataframe(sns.lineplot, x="Percentile", y="Error", hue="Model Path")#.add_legend()
#         plt.savefig(os.path.join(output_dir, f"cutoff_eps-{eps}_ood-{ood}_trial_panel.pdf"))
#         plt.show()


#     return df_cutoff


# def gen_calibration_plot(df_image, eps=0.0, ood=False, plot=True):
#     print(f"Generating calibration plot with eps={eps}, ood={ood}")
#     df = df_image[(df_image["Epsilon"]==eps) & (df_image["OOD"]==ood)]
#     # df = df.iloc[::10]
#     df_pixel = df_image_to_pixels(df, keys=["Target", "Mu", "Sigma"])

#     df_calibration = pd.DataFrame(
#         columns=["Method", "Model Path", "Expected Conf.", "Observed Conf."])

#     for method, model_path_list in trained_models.items():
#         for model_i, model_path in enumerate(tqdm(model_path_list)):

#             df_model = df_pixel[(df_pixel["Method"]==method) & (df_pixel["Model Path"]==model_path)]
#             expected_p = np.arange(41)/40.

#             observed_p = []
#             for p in expected_p:
#                 ppf = scipy.stats.norm.ppf(p, loc=df_model["Mu"], scale=df_model["Sigma"])
#                 obs_p = (df_model["Target"] < ppf).mean()
#                 observed_p.append(obs_p)

#             df_single = pd.DataFrame({'Method': method, 'Model Path': model_path,
#                 'Expected Conf.': expected_p, 'Observed Conf.': observed_p})
#             df_calibration = df_calibration.append(df_single)

#     df_truth = pd.DataFrame({'Method': Model.GroundTruth.value, 'Model Path': "",
#         'Expected Conf.': expected_p, 'Observed Conf.': expected_p})
#     df_calibration = df_calibration.append(df_truth)

#     df_calibration['Calibration Error'] = np.abs(df_calibration['Expected Conf.'] - df_calibration['Observed Conf.'])
#     df_calibration["Epsilon"] = eps
#     table = df_calibration.groupby(["Method", "Model Path"])["Calibration Error"].mean().reset_index()
#     table = pd.pivot_table(table, values="Calibration Error", index="Method", aggfunc=[np.mean, np.std, scipy.stats.sem])

#     if plot:
#         print(table)
#         table.to_csv(os.path.join(output_dir, "calib_errors.csv"))

#         print("Plotting confidence plots")
#         sns.lineplot(x="Expected Conf.", y="Observed Conf.", hue="Method", data=df_calibration)
#         plt.savefig(os.path.join(output_dir, f"calib_eps-{eps}_ood-{ood}.pdf"))
#         plt.show()

#         print(df_calibration.columns)  # 检查列名
#         print(df_calibration.head())  # 查看前几行数据

#         g = sns.FacetGrid(df_calibration, col="Method", legend_out=False)
#         # g = g.map_dataframe(sns.lineplot, x="Expected Conf.", y="Observed Conf.", hue="Model Path")#.add_legend()
#         plt.savefig(os.path.join(output_dir, f"calib_eps-{eps}_ood-{ood}_panel.pdf"))
#         plt.show()

#     return df_calibration, table



# def gen_adv_plots(df_image, ood=False):
#     print(f"Generating calibration plot with ood={ood}")
#     df = df_image[df_image["OOD"]==ood]
#     # df = df.iloc[::10]
#     df_pixel = df_image_to_pixels(df, keys=["Target", "Mu", "Sigma", "Epsilon"])
#     df_pixel["Error"] = np.abs(df_pixel["Mu"] - df_pixel["Target"])
#     df_pixel["Entropy"] = 0.5*np.log(2*np.pi*np.exp(1.)*(df_pixel["Sigma"]**2))

#     ### Plot epsilon vs error per method
#     df = df_pixel.groupby([df_pixel.index, "Method", "Model Path", "Epsilon"]).mean().reset_index()
#     df_by_method = df_pixel.groupby(["Method", "Model Path", "Epsilon"]).mean().reset_index()
#     sns.lineplot(x="Epsilon", y="Error", hue="Method", data=df_by_method)
#     plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_method_error.pdf"))
#     plt.show()

#     ### Plot epsilon vs uncertainty per method
#     sns.lineplot(x="Epsilon", y="Sigma", hue="Method", data=df_by_method)
#     plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_method_sigma.pdf"))
#     plt.show()
#     # df_by_method["Entropy"] = 0.5*np.log(2*np.pi*np.exp(1.)*(df_by_method["Sigma"]**2))
#     # sns.lineplot(x="Epsilon", y="Entropy", hue="Method", data=df_by_method)
#     # plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_method_entropy.pdf"))
#     # plt.show()


#     ### Plot entropy cdf for different epsilons
#     df_cumdf = pd.DataFrame(columns=["Method", "Model Path", "Epsilon", "Entropy", "CDF"])
#     unc_ = np.linspace(df["Entropy"].min(), df["Entropy"].max(), 100)

#     for method in df["Method"].unique():
#         for model_path in df["Model Path"].unique():
#             for eps in df["Epsilon"].unique():
#                 df_subset = df[
#                     (df["Method"]==method) &
#                     (df["Model Path"]==model_path) &
#                     (df["Epsilon"]==eps)]
#                 if len(df_subset) == 0:
#                     continue
#                 unc = np.sort(df_subset["Entropy"])
#                 prob = np.linspace(0,1,unc.shape[0])
#                 f_cdf = scipy.interpolate.interp1d(unc, prob, fill_value=(0.,1.), bounds_error=False)
#                 prob_ = f_cdf(unc_)

#                 df_single = pd.DataFrame({'Method': method, 'Model Path': model_path,
#                     'Epsilon': eps, "Entropy": unc_, 'CDF': prob_})
#                 df_cumdf = df_cumdf.append(df_single)

#     g = sns.FacetGrid(df_cumdf, col="Method")
#     g = g.map_dataframe(sns.lineplot, x="Entropy", y="CDF", hue="Epsilon", ci=None).add_legend()
#     plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_cdf_method.pdf"))
#     plt.show()

#     # NOT USED FOR THE FINAL PAPER, BUT FEEL FREE TO UNCOMMENT AND RUN
#     # ### Plot calibration for different epsilons/methods
#     # print("Computing calibration plots per epsilon")
#     # calibrations = []
#     # tables = []
#     # for eps in tqdm(df["Epsilon"].unique()):
#     #     df_calibration, table = gen_calibration_plot(df_image.copy(), eps, plot=False)
#     #     calibrations.append(df_calibration)
#     #     tables.append(table)
#     # df_calibration = pd.concat(calibrations, ignore_index=True)
#     # df_table = pd.concat(tables, ignore_index=True)
#     # df_table.to_csv(os.path.join(output_dir, f"adv_ood-{ood}_calib_error.csv"))
#     #
#     #
#     # sns.catplot(x="Method", y="Calibration Error", hue="Epsilon", data=df_calibration, kind="bar")
#     # plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_calib_error_method.pdf"))
#     # plt.show()
#     #
#     # sns.catplot(x="Epsilon", y="Calibration Error", hue="Method", data=df_calibration, kind="bar")
#     # plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_calib_error_epsilon.pdf"))
#     # plt.show()
#     #
#     # g = sns.FacetGrid(df_calibration, col="Method")
#     # g = g.map_dataframe(sns.lineplot, x="Expected Conf.", y="Observed Conf.", hue="Epsilon")
#     # g = g.add_legend()
#     # plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_calib_method.pdf"))
#     # plt.show()


# def gen_ood_comparison(df_image, unc_key="Entropy"):
#     print(f"Generating OOD plots with unc_key={unc_key}")

#     df = df_image[df_image["Epsilon"]==0.0] # Remove adversarial noise experiments
#     # df = df.iloc[::5]
#     df_pixel = df_image_to_pixels(df, keys=["Target", "Mu", "Sigma", "OOD"])
#     df_pixel["Entropy"] = 0.5*np.log(2*np.pi*np.exp(1.)*(df_pixel["Sigma"]**2))

#     df_by_method = df_pixel.groupby(["Method","Model Path", "OOD"])
#     df_by_image = df_pixel.groupby([df_pixel.index, "Method","Model Path", "OOD"])

#     df_mean_unc = df_by_method[unc_key].mean().reset_index() #mean of all pixels per method
#     df_mean_unc_img = df_by_image[unc_key].mean().reset_index() #mean of all pixels in every method and image

#     sns.catplot(x="Method", y=unc_key, hue="OOD", data=df_mean_unc_img, kind="violin")
#     plt.savefig(os.path.join(output_dir, f"ood_{unc_key}_violin.pdf"))
#     plt.show()

#     sns.catplot(x="Method", y=unc_key, hue="OOD", data=df_mean_unc_img, kind="box", whis=0.5, showfliers=False)
#     plt.savefig(os.path.join(output_dir, f"ood_{unc_key}_box.pdf"))
#     plt.show()


#     ### Plot PDF for each Method on both OOD and IN
#     g = sns.FacetGrid(df_mean_unc_img, col="Method", hue="OOD")
#     g.map(sns.distplot, "Entropy").add_legend()
#     plt.savefig(os.path.join(output_dir, f"ood_{unc_key}_pdf_per_method.pdf"))
#     plt.show()


#     ### Grab some sample images of most and least uncertainty
#     for method in df_mean_unc_img["Method"].unique():
#         imgs_max = dict()
#         imgs_min = dict()
#         for ood in df_mean_unc_img["OOD"].unique():
#             df_subset = df_mean_unc_img[
#                 (df_mean_unc_img["Method"]==method) &
#                 (df_mean_unc_img["OOD"]==ood)]
#             if len(df_subset) == 0:
#                 continue

#             def get_imgs_from_idx(idx):
#                 i_img = df_subset.loc[idx]["level_0"]
#                 img_data = df_image.loc[i_img]
#                 sigma = np.array(img_data["Sigma"])
#                 entropy = np.log(sigma**2)

#                 ret = [img_data["Input"], img_data["Mu"], entropy]
#                 return list(map(trim, ret))

#             def idxquantile(s, q=0.5, *args, **kwargs):
#                 qv = s.quantile(q, *args, **kwargs)
#                 return (s.sort_values()[::-1] <= qv).idxmax()

#             imgs_max[ood] = get_imgs_from_idx(idx=idxquantile(df_subset["Entropy"], 0.95))
#             imgs_min[ood] = get_imgs_from_idx(idx=idxquantile(df_subset["Entropy"], 0.05))

#         all_entropy_imgs = np.array([ [d[ood][2] for ood in d.keys()] for d in (imgs_max, imgs_min)])
#         entropy_bounds = (all_entropy_imgs.min(), all_entropy_imgs.max())

#         Path(os.path.join(output_dir, "images")).mkdir(parents=True, exist_ok=True)
#         for d in (imgs_max, imgs_min):
#             for ood, (x, y, entropy) in d.items():
#                 id = os.path.join(output_dir, f"images/method_{method}_ood_{ood}_entropy_{entropy.mean()}")
#                 cv2.imwrite(f"{id}_0.png", 255*x)
#                 cv2.imwrite(f"{id}_1.png", apply_cmap(y, cmap=cv2.COLORMAP_JET))
#                 entropy = (entropy - entropy_bounds[0]) / (entropy_bounds[1]-entropy_bounds[0])
#                 cv2.imwrite(f"{id}_2.png", apply_cmap(entropy))



#     ### Plot CDFs for every method on both OOD and IN
#     df_cumdf = pd.DataFrame(columns=["Method", "Model Path", "OOD", unc_key, "CDF"])
#     unc_ = np.linspace(df_mean_unc_img[unc_key].min(), df_mean_unc_img[unc_key].max(), 200)

#     for method in df_mean_unc_img["Method"].unique():
#         for model_path in df_mean_unc_img["Model Path"].unique():
#             for ood in df_mean_unc_img["OOD"].unique():
#                 df = df_mean_unc_img[
#                     (df_mean_unc_img["Method"]==method) &
#                     (df_mean_unc_img["Model Path"]==model_path) &
#                     (df_mean_unc_img["OOD"]==ood)]
#                 if len(df) == 0:
#                     continue
#                 unc = np.sort(df[unc_key])
#                 prob = np.linspace(0,1,unc.shape[0])
#                 f_cdf = scipy.interpolate.interp1d(unc, prob, fill_value=(0.,1.), bounds_error=False)
#                 prob_ = f_cdf(unc_)

#                 df_single = pd.DataFrame({'Method': method, 'Model Path': model_path,
#                     'OOD': ood, unc_key: unc_, 'CDF': prob_})
#                 df_cumdf = df_cumdf.append(df_single)

#     sns.lineplot(data=df_cumdf, x=unc_key, y="CDF", hue="Method", style="OOD")
#     plt.savefig(os.path.join(output_dir, f"ood_{unc_key}_cdfs.pdf"))
#     plt.show()



# if args.load_pkl:
# # if 1:
#     print("Loading!")
#     df_image = pd.read_pickle("cached_depth_results.pkl")
# else:
#     df_image = compute_predictions()
#     df_image.to_pickle("cached_depth_results.pkl")

# Path(output_dir).mkdir(parents=True, exist_ok=True)
#     # 可视化函数 gen_cutoff_plot(df_image) 等需要适配 PyTorch 版本
# gen_cutoff_plot(df_image)
# gen_calibration_plot(df_image)
# gen_adv_plots(df_image)
# gen_ood_comparison(df_image)