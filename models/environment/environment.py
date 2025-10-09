import torch
from typing import List
from ..util import transform
from typing import Optional, List
from contextlib import contextmanager
class Env:
    def __init__(self, 
            max_deg:Optional[List[float]]=None,
            min_deg:Optional[List[float]]=None,
            max_tran:Optional[List[float]]=None,
            min_tran:Optional[List[float]]=None,
            mag_randomly: bool = True,
            **argv) -> None:
        
        self.max_deg = max_deg
        self.min_deg = min_deg
        self.max_tran = max_tran
        self.min_tran = min_tran
        self.mag_randomly = mag_randomly
        self.transform_class = transform.UniformTransformSE3(self.max_deg, self.max_tran, self.min_deg, self.min_tran, self.mag_randomly)
        self.tsl_transform_class = transform.UniformTransformSE3(0, self.max_tran, 0, self.min_tran, self.mag_randomly)
        self.rot_transform_class = transform.UniformTransformSE3(self.max_deg, 0, self.min_deg, 0, self.mag_randomly)  # actually, tsl mlp does not require such large roatation perturbation.

    def __repr__(self):
        return f"max_deg: {self.max_deg}, min_deg: {self.min_deg}, max_tran: {self.max_tran}, min_tran: {self.min_tran}, mag_randomly: {self.mag_randomly}."

    def set_mag_attr(self, **argv):
        for key, value in argv.items():  # support partial assignment
            setattr(self, key, value)
        self.transform_class = transform.UniformTransformSE3(self.max_deg, self.max_tran, self.min_deg, self.min_tran, self.mag_randomly)
        self.tsl_transform_class = transform.UniformTransformSE3(0, self.max_tran, 0, self.min_tran, self.mag_randomly)
        self.rot_transform_class = transform.UniformTransformSE3(self.max_deg, 0, self.min_deg, 0, self.mag_randomly)

    @contextmanager
    def set_mag_attr_temp(self, **argv):
        """leverage buffer provided by itself

        Args:
            img (torch.Tensor): (B, 3, H, W)
            pcd (torch.Tensor): (B, N, 3)
        """
        tmp_argv = dict()
        try:
            for key in argv.keys():  # support partial assignment
                tmp_argv[key] = getattr(self, key)
            self.set_mag_attr(**argv)
            yield
        finally:
            self.set_mag_attr(**tmp_argv)  # store original parameters

    def perturb(self, extran: torch.Tensor, num_perturbations: int, disentangled: bool = True,
                concat_input: bool = False, add_rot_perturb: bool = True, add_tsl_perturb: bool = True,
                add_rot_perturb_to_gt: bool = False, add_tsl_perturb_to_gt: bool = False) -> torch.Tensor:
        """
        add perturb to extran (gt) (B, 4, 4) -> (B, G, 4, 4) with (B, 0, 4, 4) as the GT one
        if concat_input is True, set (B, 0, 4, 4) as the GT matrix
        """
        # observation
        B = extran.shape[0]  # GT Tcl
        if concat_input:
            if add_rot_perturb_to_gt and add_tsl_perturb_to_gt:
                G = num_perturbations - 2
            else:
                G = num_perturbations - 1
        else:
            G = num_perturbations
        if add_rot_perturb and add_tsl_perturb:
            perturb = self.transform_class.generate_transform(B * G, return_se3=True).to(extran)  # (B * G, 4, 4)
        elif add_rot_perturb:
            perturb = self.rot_transform_class.generate_transform(B * G, return_se3=True).to(extran)  # (B * G, 4, 4)
        elif add_tsl_perturb:
            perturb = self.tsl_transform_class.generate_transform(B * G, return_se3=True).to(extran)  # (B * G, 4, 4)
        else:
            raise NotImplementedError("At least one of add_rot_perturb and add_tsl_perturb should be set to True")
        perturb = perturb.view(B, G, 4, 4)
        if concat_input:
            if (not add_rot_perturb_to_gt) and (not add_tsl_perturb_to_gt):
                gt_mat = torch.eye(4).view(1, 1, 4, 4).expand(B, -1, -1, -1).to(perturb)
            elif add_rot_perturb_to_gt and (not add_tsl_perturb_to_gt):
                gt_mat = self.rot_transform_class.generate_transform(B, return_se3=True).view(B, 1, 4, 4).to(extran)
            elif add_tsl_perturb_to_gt and (not add_rot_perturb_to_gt):
                gt_mat = self.tsl_transform_class.generate_transform(B, return_se3=True).view(B, 1, 4, 4).to(extran)
            else:  # gt_rot_perturb and gt_tsl_perturb are both True, use tow gt matrices
                gt_mat = torch.cat([
                    self.tsl_transform_class.generate_transform(B, return_se3=True).view(B, 1, 4, 4).to(extran),
                    self.rot_transform_class.generate_transform(B, return_se3=True).view(B, 1, 4, 4).to(extran)],
                    dim=1)  # (B, 2, 4, 4)
            perturb = torch.cat([gt_mat, perturb], dim=1)  # (B, 1, 4, 4), (B, G-1, 4, 4) -> (B, G, 4, 4)
        if not disentangled:
            return perturb @ extran.view(B, 1, 4, 4).expand(-1, num_perturbations, -1, -1)  # repeat at dim=1
        else:
            res = extran.view(B, 1, 4, 4).repeat(1, num_perturbations, 1, 1)  # (B, G, 4, 4)
            if add_rot_perturb:
                res[..., :3, :3] = perturb[..., :3, :3] @ res[..., :3, :3]
            if add_tsl_perturb:
                res[..., :3, 3] = perturb[..., :3, 3] + res[..., :3, 3]
            return res