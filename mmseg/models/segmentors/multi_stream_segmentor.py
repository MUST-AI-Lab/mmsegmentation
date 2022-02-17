from .base import BaseSegmentor


class MultiStreamSegmentor(BaseSegmentor):
    """This is a wrapper for muli-stream segmentor, where there are multiple update streams when training.
    This architecture is mainly used in semi-supervised segmentation( student-teacher types ).
    The following codes are modified based on the implementation of SoftTeacher model.
    """

    def __init__(
            self, model, train_cfg=None, test_cfg=None
    ):  # model: a dict containing all branches (e.g., teacher, student).
        super(MultiStreamSegmentor, self).__init__()
        self.branches = list(model.keys())
        for k, v in model.items():
            setattr(self, k, v)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.test_cfg:
            self.inference_on = self.test_cfg.get("inference_on", self.branches[0])
        else:
            self.inference_on = self.branches[0]
            
        # select a branch used for prediction

    def model(self, **kwargs):
        """
        This method returns the branch used for inference, not really useful though.

        Args:
            **kwargs: include branch==[branch name], ok if not included

        Returns:
            inference branch or the specified branch
        """
        if "branch" in kwargs:
            assert (
                    kwargs["branch"] in self.branches
            ), "Detector does not contain submodule {}".format(kwargs["branch"])
            model = getattr(self, kwargs["branch"])
        else:
            model = getattr(self, self.inference_on)
        return model

    def forward_test(self, imgs, img_metas, **kwargs):

        return self.model(**kwargs).forward_test(imgs, img_metas, **kwargs)

    def extract_feat(self, imgs):
        return self.model().extract_feat(imgs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.model().aug_test(imgs, img_metas, kwargs)

    def encode_decode(self, img, img_metas):
        return self.model().encode_decode(img, img_metas)

    def simple_test(self, img, img_meta, **kwargs):
        return self.model().simple_test(img, img_meta, kwargs)

    def init_weights(self):
        for key in self.branches:
            branch = getattr(self, key)
            branch.init_weights()
