import argparse
import json
import os
import random
import string
import subprocess

import pykrylov
import pykrylov.ems as experiment
from pykrylov.util.consts import EXP_ID

MASTER_PORT = 2020
PVCS = {
    "iu-pvc": "krylov-user-pvc-image-understanding",
    "retina-pvc": "krylov-user-pvc-retina",
    "nushare2": "krylov-user-pvc-nlp-01",
    "nushare": "krylov-user-pvc-nlp-45",
    "mtrepo": "nlp-ebert-01",
}
DOCKER_IMAGE_TAG = "4eceabfb0592b0a4a15f053012af505617191bdc"


def get_file_path(file_path: str) -> str:
    """Get the file path in the Krylov environment"""
    if "KRYLOV_WF_HOME" in os.environ:
        task_folder = os.path.join(
            os.environ["KRYLOV_WF_HOME"], "src", os.environ["KRYLOV_WF_TASK_NAME"]
        )
        file_name = os.path.basename(file_path)
        return os.path.join(task_folder, file_name)
    else:
        return file_path


def init_krylov_common_context(pvc_mount_name: str):
    krylov_data_dir = os.path.join("/mnt", pvc_mount_name, "lmms-eval/cache")
    os.environ["HF_HOME"] = os.path.join(krylov_data_dir, ".lmm_cache/hf_cache")
    os.environ["HF_MODULES_CACHE"] = os.path.join(
        krylov_data_dir, ".lmm_cache/hf_module_cache"
    )
    os.environ["HF_DATASETS_CACHE"] = os.path.join(
        krylov_data_dir, ".lmm_cache/tmp/hf_data_cache"
    )
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(
        krylov_data_dir, ".lmm_cache/hf_cache"
    )

    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(
        krylov_data_dir, ".lmm_cache/torch_cache"
    )
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        krylov_data_dir, ".lmm_cache/.triton"
    )
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(
        krylov_data_dir, ".lmm_cache/.numba"
    )
    os.environ["OUTLINES_CACHE_DIR"] = os.path.join(
        krylov_data_dir, ".lmm_cache/.outlines"
    )
    os.environ["PVC_MOUNT_PATH"] = os.path.join("/mnt", pvc_mount_name)


def init_krylov_workspace_context(pvc_mount_name: str):
    os.environ["PVC_MOUNT_PATH"] = os.path.join("/mnt", pvc_mount_name)
    os.environ["LMMS_EVAL_LOG_DIR"] = "logs"


def set_hf_offline():
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_EVALUATE_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def init_krylov_context(pvc_mount_name: str):
    init_krylov_common_context(pvc_mount_name)
    if "KRYLOV_WS_NAME" not in os.environ:
        # Get params
        context = pykrylov.util.get_task_context()
        if "experiment_id" in context:
            experiment_id = context["experiment_id"]
            # These 2 lines make task logs viewable via experiment view on aihub
            pykrylov.util.set_global_context({EXP_ID: experiment_id})
            pykrylov.ems.experiment.update_experiment(
                experiment_id,
                runtime={"workflow": {"runId": os.environ["KRYLOV_WF_RUN_ID"]}},
            )

        os.environ["LMMS_EVAL_LOG_DIR"] = os.path.join(
            os.environ["KRYLOV_WF_HOME"],
            "src",
            os.environ["KRYLOV_WF_TASK_NAME"],
            "logs",
        )

        return {
            "gpu_per_node": int(context["gpu_per_node"]),
            "num_nodes": int(context["num_nodes"]),
        }


def get_distributed_info():
    context = pykrylov.util.get_task_context()
    master_name = context["master_name"]
    master_service_name = context["master_service_name"]
    # Get master IP address from krylov environment variables
    print(f"Getting a list of ip addresses for of service {master_name}")
    ip_list = pykrylov.distributed.get_ip_list(master_name, master_service_name)
    print(f"Got a list of ips: {' '.join(ip_list)}")
    master_addr = ip_list[0]  # master rank always 0
    # Get rank from krylov task index environment variable
    print("Getting rank of this task")
    rank = pykrylov.distributed.get_task_index()
    print(f"Rank of this task is {rank}")
    return {
        "ip_list": ip_list,
        "master_addr": master_addr,
        "rank": rank,
    }


def run_fn(script, model_name, model_args, lmms_eval_task, pvc_mount_name, hf_offline=True):
    if hf_offline:
        set_hf_offline()
    context = init_krylov_context(pvc_mount_name)
    if context["num_nodes"] > 1:
        dist_info = get_distributed_info()
        os.environ["MASTER_ADDR"] = dist_info["master_addr"]
        os.environ["NODE_RANK"] = str(dist_info["rank"])

    script_path = get_file_path(script)
    os.chmod(script_path, 0o755)
    output = subprocess.run([script_path, model_name, model_args, lmms_eval_task], check=True)

    log_metrics()
    record_results_as_assets()

    if output.returncode != 0:
        raise ValueError(f"Script exited with error {output.returncode}")

def run_fn_local(script_path, model_name, model_args, lmms_eval_task, pvc_mount_name, hf_offline=True):
    if hf_offline:
        set_hf_offline()
    init_krylov_workspace_context(pvc_mount_name)
    os.chmod(script_path, 0o755)
    output = subprocess.run([script_path, model_name, model_args, lmms_eval_task], check=True)
    if output.returncode != 0:
        raise ValueError(f"Script exited with error {output.returncode}")


def log_metrics():
    eval_log_dir = os.environ["LMMS_EVAL_LOG_DIR"]

    log_dir = os.path.join(eval_log_dir, next(iter(os.listdir(eval_log_dir))))
    for file in os.listdir(log_dir):
        if file.endswith("results.json"):
            results_file = os.path.join(log_dir, file)
            with open(results_file) as f:
                results = json.load(f)
            break

    task_metrics = next(iter(results["results"].values()))

    for metric, value in task_metrics.items():
        if "stderr" not in metric:
            experiment.record_metric(metric.split(",")[0], value)


def record_results_as_assets():
    eval_log_dir = os.environ["LMMS_EVAL_LOG_DIR"]
    log_dir = os.path.join(eval_log_dir, next(iter(os.listdir(eval_log_dir))))
    for file in os.listdir(log_dir):
        if file.endswith(".json"):
            experiment.record_asset(file, os.path.join(log_dir, file))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", help="Which script to run")
    parser.add_argument("--model_name", default="llava_hf", help="LMM name")
    parser.add_argument("--model_args", default="", help="LMM model arguments")
    parser.add_argument("--lmms_eval_task", default="regulatory_doc", help="LMMS eval task name")
    parser.add_argument(
        "--experiment_name", default="lmms-eval", help="Experiment name"
    )
    parser.add_argument(
        "--ems_project", default="image-understanding", help="EMS project name"
    )
    parser.add_argument("--cluster", default="tess38", help="Krylov cluster")
    parser.add_argument("-n", "--namespace", default="coreai-iu-lvs-a100", type=str)
    parser.add_argument(
        "-i",
        "--image",
        default=f"hub.tess.io/image-understanding/lmms-eval:{DOCKER_IMAGE_TAG}",
        help="Docker image",
    )
    parser.add_argument("--cpu", default=8, help="Use cpus")
    parser.add_argument("--gpu", default="a100", type=str, help="GPU model to use (a100 or v100)")
    parser.add_argument("--memory", default=32, help="Use memory")
    parser.add_argument(
        "--gpu_per_node", default=1, type=int, help="How many GPUs per node"
    )
    parser.add_argument("--num_nodes", default=1, type=int, help="How many nodes")
    parser.add_argument("--pvc", default="retina-pvc", help="PVC name to mount, default to retina-pvc")
    parser.add_argument("--allow_hf_connection", action="store_true", default=False, help="Turns off HF offline mode, meaning you could download models from HF hub")

    # Not submit_task related
    parser.add_argument("--local", action="store_true", help="Execute locally instead of submitting experiment")
    return parser.parse_args()


def submit_task(
    script,
    model_name,
    model_args,
    lmms_eval_task,
    experiment_name,
    ems_project,
    cluster,
    namespace,
    image,
    cpu,
    gpu,
    memory,
    gpu_per_node,
    num_nodes,
    pvc,
    allow_hf_connection
):
    master_name = "lmms_" + "".join(random.choices(string.ascii_letters, k=8))
    master_service_name = master_name + "_svc"

    if num_nodes > 1:
        task = pykrylov.distributed.DistributedTask(
            run_fn,
            args=[
                os.path.basename(script),
                model_name,
                model_args,
                lmms_eval_task,
                pvc,
                not allow_hf_connection,
            ],
            docker_image=image,
            parallelism=num_nodes,
            name=master_name,
            service_name=master_service_name,
            service_port=MASTER_PORT,
        )
    else:
        task = pykrylov.Task(
            run_fn,
            args=[
                os.path.basename(script),
                model_name,
                model_args,
                lmms_eval_task,
                pvc,
                not allow_hf_connection,
            ],
            docker_image=image,
        )

    task.add_task_parameters(
        {
            "ems_project": ems_project,
            "experiment_name": experiment_name,
            "gpu_per_node": gpu_per_node,
            "num_nodes": num_nodes,
            "master_name": master_name,
            "master_service_name": master_service_name,
            "master_port": MASTER_PORT,
        }
    )

    task.add_cpu(cpu)
    task.add_memory(memory)
    if gpu_per_node > 0:
        task.run_on_gpu(quantity=gpu_per_node, model=gpu)

    if pvc in PVCS.keys():
        task.mount_pvc(pvc, PVCS[pvc], cluster)
    else:
        raise ValueError(f"Unknown PVC {pvc}")

    task.add_file(script)

    workflow = pykrylov.Flow(task)
    if cluster in ["tess94", "tess137", "tess38"]:
        workflow.execution_parameters.add_execution_parameter("enableChooseCluster", "true")

    session = pykrylov.Session(namespace=namespace)
    run_id = session.submit_experiment(
        workflow,
        project=ems_project,
        experiment_name=experiment_name,
        labels=["lmms-eval"],
    )

    print(f"Submitted a task with run_id {run_id}")
    if run_id:
        message = pykrylov.ems.attributes.add_runtime(
            run_id,
            {
                "workflow": {
                    "runId": run_id,
                }
            },
        )
        print(message)

        link = f"https://aip.vip.ebay.com/data/experiment-detail?projectName={ems_project}&experimentId={run_id}"
        print(f"You can monitor progress and download result by visiting {link}")


def main():
    args = parse_args()
    print(args)

    if args.local:
        run_fn_local(
            args.script,
            args.model_name,
            args.model_args,
            args.lmms_eval_task,
            args.pvc,
            not args.allow_hf_connection,
        )
    else:
        submit_task(
            args.script,
            args.model_name,
            args.model_args,
            args.lmms_eval_task,
            args.experiment_name,
            args.ems_project,
            args.cluster,
            args.namespace,
            args.image,
            args.cpu,
            args.gpu,
            args.memory,
            args.gpu_per_node,
            args.num_nodes,
            args.pvc,
            args.allow_hf_connection,
        )


if __name__ == "__main__":
    main()
